//! The `rmcp` server handler: catalog, dispatch, the response budget, and
//! the SEP-2663 tasks extension for the sweep.

use std::sync::Arc;

use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CancelTaskParams, CreateTaskResult, GetTaskParams,
    GetTaskResult, Implementation, ListToolsResult, PaginatedRequestParams, ServerCapabilities,
    ServerConfig, Tool, UpdateTaskParams,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::task_manager::{TaskExit, TaskOptions};
use rmcp::{ErrorData as McpError, ServerHandler, ServiceExt};

use crate::envelope::{result_size_bytes, too_large};
use crate::guide::INSTRUCTIONS;
use crate::jobs::ProgressTarget;
use crate::state::ServerState;
use crate::tools::{self, CallContext, ToolSpec};

#[derive(Clone)]
pub struct PulsarServer {
    state: Arc<ServerState>,
    specs: Arc<Vec<ToolSpec>>,
    catalog: Arc<Vec<Tool>>,
}

impl PulsarServer {
    pub fn new(state: Arc<ServerState>) -> Self {
        let specs = tools::catalog();
        let catalog = specs.iter().map(ToolSpec::to_tool).collect();
        Self {
            state,
            specs: Arc::new(specs),
            catalog: Arc::new(catalog),
        }
    }

    pub fn state(&self) -> &Arc<ServerState> {
        &self.state
    }

    /// The advertised tools, in order.
    pub fn tools(&self) -> &[Tool] {
        &self.catalog
    }

    fn known(&self, name: &str) -> bool {
        self.specs.iter().any(|s| s.name == name)
    }

    async fn invoke(
        state: Arc<ServerState>,
        name: String,
        params: CallToolRequestParams,
        progress: Option<ProgressTarget>,
    ) -> rmcp::model::CallToolResult {
        let args = params.arguments.unwrap_or_default();
        let result =
            match tools::dispatch(state.clone(), &name, args, CallContext { progress }).await {
                Ok(result) => result,
                Err(error) => error.into_result(),
            };
        match state.options.max_response_bytes {
            Some(limit) if result_size_bytes(&result) > limit => {
                too_large(&name, result_size_bytes(&result), limit)
            }
            _ => result,
        }
    }
}

impl ServerHandler for PulsarServer {
    fn get_info(&self) -> ServerConfig {
        let capabilities = ServerCapabilities::builder()
            .enable_tools()
            .enable_tasks()
            .build();
        let mut info = ServerConfig::new(capabilities);
        let mut implementation =
            Implementation::new("pulsar", env!("CARGO_PKG_VERSION")).with_title("Pulsar");
        implementation.website_url = Some("https://github.com/Krv-Labs/pulsar".into());
        info.server_info = implementation;
        info.instructions = Some(INSTRUCTIONS.to_owned());
        info
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(self.catalog.to_vec()))
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        self.catalog.iter().find(|t| t.name == name).cloned()
    }

    async fn call_tool(
        &self,
        params: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let name = params.name.to_string();
        if !self.known(&name) {
            return Err(McpError::invalid_params(
                format!("unknown tool {name:?}"),
                None,
            ));
        }
        let progress = context
            .meta
            .get_progress_token()
            .map(|token| ProgressTarget {
                peer: context.peer.clone(),
                token,
            });
        let wants_task = name == tools::sweep::SWEEP
            && context
                .client_capabilities()
                .is_some_and(|c| c.supports_tasks())
            && params
                .arguments
                .as_ref()
                .and_then(|a| a.get("execution"))
                .and_then(|v| v.as_str())
                != Some("async");
        if wants_task {
            let state = self.state.clone();
            let task = self.state.tasks.spawn(
                TaskOptions::new()
                    .with_status_message("sweep queued")
                    .with_poll_interval_ms(500),
                move |task_ctx| {
                    Box::pin(async move {
                        task_ctx.set_status_message("sweep running");
                        let result = Self::invoke(state, name, params, progress).await;
                        if task_ctx.is_cancel_requested() {
                            return Err(TaskExit::Cancelled);
                        }
                        Ok(result)
                    })
                },
            );
            return Ok(CallToolResponse::Task(CreateTaskResult::new(task)));
        }
        let result = Self::invoke(self.state.clone(), name, params, progress).await;
        Ok(CallToolResponse::Complete(result))
    }

    async fn get_task(
        &self,
        request: GetTaskParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<GetTaskResult, McpError> {
        self.state
            .tasks
            .get_task(&request.task_id)
            .map(GetTaskResult::new)
    }

    async fn update_task(
        &self,
        request: UpdateTaskParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<(), McpError> {
        self.state
            .tasks
            .update_task(&request.task_id, request.input_responses)
    }

    async fn cancel_task(
        &self,
        request: CancelTaskParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<(), McpError> {
        self.state.tasks.cancel_task(&request.task_id)
    }
}

/// Serve on stdio until the client disconnects.
pub async fn run_stdio(
    state: Arc<ServerState>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let service = PulsarServer::new(state)
        .serve(rmcp::transport::stdio())
        .await?;
    service.waiting().await?;
    Ok(())
}
