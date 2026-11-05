//! `outcome_columns`: the framing question asked when no target was named.

use crate::plan::defaults::DraftContext;
use crate::plan::model::{PendingQuestion, PlanQuestion, QuestionOption, QuestionTopic};

pub(super) fn build(ctx: &mut DraftContext) -> Vec<PendingQuestion> {
    if !ctx.target_set.is_empty() {
        return vec![];
    }
    let question = PlanQuestion::new(
        QuestionTopic::OutcomeColumns,
        "Is any column a result you want to explain after the map is drawn (for example a diagnosis, outcome, or survival status)? Columns used to draw the map cannot fairly be used to explain it afterwards.".into(),
        vec![],
        ctx.n_rows,
        vec![QuestionOption::new(
            "none_of_them",
            "None of them",
            "Keeps all current columns as map inputs",
            vec![],
        )],
        "none_of_them",
    );
    vec![(question, vec![])]
}
