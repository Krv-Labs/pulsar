# Pulsar MCP: The Domain Expert's Guide to Topological Discovery

Until now, trying to find complex, high-dimensional shapes in your datasets required an annoying amount of custom Python code, a PhD in algebraic topology, and fighting with K-Means until you wanted to throw your laptop out a window.

By default, traditional clustering just forces your data into neat little spheres—even when your data _really is_ a weird, continuous, sprawling manifold. It can be surprising and unintuitive.

The **Pulsar MCP Server** is our attempt to give you what you _actually_ want. It gives your favorite AI (like Claude or Gemini) a set of "Thick Tools" to map the geometric structure of your data. You just point it at a CSV and tell it to figure out the shape of reality.

What follows from here is a walkthrough to help you—the domain expert—actually use this thing, using the classic Palmer Penguins dataset as a dummy example to show you how to eventually do this on your own data.

> [!TIP]
> **Who is this for?**
> If you know what your data _means_ but don't want to write a custom scikit-learn pipeline, this is for you. We handle the math; you handle the science.

---

## The Setup (Getting the plumbing working)

Before we start mapping the cosmos, we need to turn the bridge on. Pulsar uses the **Model Context Protocol (MCP)**. Think of this as a secure "bridge" that lets your AI safely talk to your data and use our topological tools.

Configuring Python environments by hand is a nightmare that belongs in the past. We highly recommend using `uv` (the blazing-fast Python package manager) to run the server. If you're still manually activating virtual environments... who hurt you?

Here is exactly how to wire up Pulsar to your AI of choice. Pick your poison:

<details>
  <summary><strong>Option A: Claude Desktop (The Visual App)</strong></summary>

  <br>

**1. Locate your Configuration File**
You need to tell Claude where the Pulsar "bridge" is. Open your file explorer and go here:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

**2. Add the Pulsar Bridge**
Open that file in a text editor (like Notepad or TextEdit) and add this block inside the `"mcpServers"` section.

```json
{
  "mcpServers": {
    "pulsar": {
      "command": "uv",
      "args": ["run", "--group", "mcp", "pulsar-mcp"]
    }
  }
}
```

> [!CAUTION]
> **Pathing Issues:** If Claude complains it can't find `uv`, replace `"command": "uv"` with the absolute path to your `uv` executable (e.g., `/Users/yourname/.cargo/bin/uv`).

**3. Restart Claude**
Fully quit Claude Desktop and restart it. Look for the little **Hammer icon** in a new chat to confirm the tools loaded. You are officially a topological cartographer.

</details>

<details>
  <summary><strong>Option B: Gemini CLI (The Fast Way)</strong></summary>

  <br>

If you're using the Gemini CLI, you literally just run one command in your terminal from the Pulsar directory:

```bash
gemini mcp add pulsar uv run --group mcp pulsar-mcp
```

That's it. You're done. Go analyze some data.

</details>

<details>
  <summary><strong>Option C: Claude Code (Anthropic's CLI)</strong></summary>

  <br>

Using Anthropic's terminal client? It's just as easy as Gemini. Run this from your terminal:

```bash
claude mcp add pulsar -- uv run --group mcp pulsar-mcp
```

</details>

<details>
  <summary><strong>Option D: Cursor / Windsurf (AI Code Editors)</strong></summary>

  <br>

If you live inside an AI IDE like Cursor (which uses OpenAI/Claude models under the hood), you don't even need to touch a config file.

1. Open **Settings** (usually `Cmd/Ctrl + Shift + J`).
2. Navigate to **Features** > **MCP**.
3. Click **+ Add new MCP server**.
4. Set the name to `pulsar`.
5. Set the type to `command`.
6. Set the command to: `uv run --group mcp pulsar-mcp`
7. Save and refresh. You'll see the Pulsar tools light up with a green dot.
</details>

---

## How to Talk to the AI (The Workflow)

Please don't just dump a CSV into an AI and say "analyze this"... you animals. That's a great way to get hallucinated garbage and an AI that writes fake python scripts that error out.

Instead, when using the Pulsar MCP server, you want to point the AI at a goal. The AI already knows the technical steps (characterization, sweeps, dossiers)—your job is to ask for **insight**.

> [!NOTE]  
> **The Golden Prompt**
> Just tell the AI: _"I have a dataset at `path/to/data.csv`. Use Pulsar to find the hidden structure and tell me the story of this data. I'm looking for meaningful subgroups and the specific traits that define them."_

### What the AI will do (The "Black Box"):

Even though you're asking for insight, the AI is doing some heavy lifting under the hood to ensure that insight is grounded in math, not hallucinations:

1. **Geometric Grounding:** It probes the data's "density" first so it doesn't just guess at parameters.
2. **Topological Mapping:** It runs a "sweep" to find the most stable version of your data's shape.
3. **Automated Iteration:** If the results look like a giant uninformative blob or a shattered mess, it will automatically tune the scales until it finds the "signal."
4. **Insight Synthesis:** It generates a **Dossier**—a deep statistical report on the subpopulations it found.

---

## Dogfooding on Flightless Birds (The Penguin Demo)

We ran this exact workflow on the Palmer Penguins dataset. Our goal? See if the geometry could naturally separate the penguin species _without us actually giving it the species labels_.

<details>
  <summary><strong>See the config that magically worked</strong></summary>

  <br>
  
  After a couple of iterations, the AI landed on this beautiful 5-dimensional setup:

```yaml
run:
  name: penguin_species_recovery_dim5
  data: "demos/MCP-penguins/penguins.csv"
preprocessing:
  drop_columns: ["species", "rowid", "year"] # Look ma, no labels!
  encode:
    island: { method: one_hot }
    sex: { method: one_hot }
  impute:
    bill_length_mm: { method: fill_mean }
    # ... you get the idea
sweep:
  pca:
    dimensions:
      values: [5]
  ball_mapper:
    epsilon:
      range:
        min: 0.80
        max: 1.50
        steps: 15
cosmic_graph:
  threshold: auto
```

</details>

### What it found

It turns out, geometry is pretty important if you don't want your analysis to look like trash. The graph shattered into distinct components that perfectly recovered the biology:

> [!IMPORTANT]
> **The biological reality**
> The topology revealed that **habitat (Island)** and **biological sex** are just as dominant structurally as the actual species.

- **The Gentoos:** Completely isolated themselves on Biscoe Island, further splitting perfectly by male and female. (They are chunky birds).
- **The Adelies:** Splintered into different groups strictly based on which island they lived on.
- **The Chinstraps:** Got lumped entirely into the Dream Island Adelies. Why? Because geometrically, a Chinstrap and an Adelie living on Dream Island share the exact same morphological envelope. The math doesn't lie.

---

## Bringing Your Own Data

Ready to stop messing with penguins and analyze your own spreadsheets?

1. **Clean your paths:** Make sure your CSV is accessible on your machine.
2. **Boot the server:** Hook the MCP server into your AI client using the setup steps above.
3. **Ask the AI:** Say, _"Hey, look at `my_amazing_data.csv` using Pulsar. I want to see if there are any hidden structural groups."_

The AI will handle the messy data imputation, the categorical encoding, and the dimension scaling. Your job is to read the Dossier at the end and say, _"Ah yes, exactly as I suspected."_

Go build something beautiful.
