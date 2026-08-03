<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin-right: 20px; height: 55px" height="55px">

# Technical Setup: Agentic Ops & Retrieval-Augmented Generation (RAG) in Practice

For all live sessions and interactive exercises, we recommend:

- **A Google account** you can sign into (for Colab)
- **A modern web browser** — Chrome recommended for best Colab compatibility, though Firefox, Edge, and Safari all work
- Access to the course's GitHub repository URL (a GitHub account is optional — see below)
- A stable internet connection for the full duration of each session (Colab is a hosted service; unlike a locally-installed Jupyter setup, it does not work offline)

We have shared separate instructions [here](https://docs.google.com/document/d/1KRCpyQjjqmot3FMT1Luhm6kmT_aYBrQoBSJ12i8tsMU/edit?usp=sharing) for setting up your api keys in advance of the class start. Please verify your keys work using the `00_Evironment_Verification.ipynb` notebook in colab. 

---

## Hardware & Operating System

Because all computation happens on Google's servers, **your local hardware barely matters**. Any device that can run a modern web browser reliably is sufficient:

| Requirement | Detail |
|---|---|
| Operating system | Any — Windows, macOS, Linux, or ChromeOS all work identically, since only the browser matters |
| Local CPU/RAM | No specific requirement; a device that can smoothly run a modern browser with a few tabs open is enough |
| Local disk space | Negligible — nothing from this course installs locally. A few hundred MB of free space is a safe margin if you also clone the repository (see below) |
| Screen size | 13" or larger recommended — you'll typically want the instructor's shared screen and your own Colab tab visible at once; a smaller screen makes that awkward, not impossible |
| GPU | Not required. Every course notebook uses `faiss-cpu`; when creating a Colab runtime, leave the hardware accelerator set to **None/CPU** (Runtime → Change runtime type) |

**Administrator privileges are not required on your laptop.** Nothing in this course is installed outside of Colab's own temporary cloud environment.

---

## Accounts You'll Need

### Google Account (required)

Colab requires signing in with a Google account. Any personal or work Google account works — no Colab Pro subscription is needed; the free tier's default CPU runtime and memory allocation comfortably cover every notebook in this course.

### GitHub Account (optional)

You do **not** need a GitHub account to access or run the course materials — the repository is public, and Colab can open notebooks directly from a public GitHub URL without you signing into GitHub at all. A GitHub account is only useful if you want to:
- Fork the repository to save your own edits back to GitHub
- Star/watch the repository for updates
- Clone it to a local machine for offline editing outside of Colab 

---

## Getting the Course Materials into Colab

You have two supported paths — pick whichever is more convenient. Neither requires installing git as a program on your computer.

### Path A — Open Notebooks Directly from GitHub (recommended for live sessions)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Open notebook → GitHub** tab
3. Paste the course repository URL
4. Select the notebook for the current module (e.g., `notebooks/01_architecting_pipelines_lab.ipynb`)
5. Colab opens a live, editable copy in your browser — no local download step at all

This is the fastest path and is what we'll use during live sessions. Each notebook's own setup cell handles installing `faiss-cpu`/`numpy`/`matplotlib`, and the Module 4/5 notebooks that load shared data files (`golden_dataset_module4.json`, `capstone_corpus.json`) include a built-in fallback that embeds the data directly in the notebook if it can't find the file on disk — so this path works even without a full repository clone.

### Path B — Clone the Repository, Then Upload

If you'd rather have the entire repository (all module guides, solutions docs, and data files, not just one notebook at a time) available in one place:

```bash
git clone https://github.com/<org>/<repo-name>.git
```

Then, inside a Colab notebook, either:
- **Upload individual files** via the Colab file browser (folder icon in the left sidebar → upload icon), or
- **Clone directly inside the Colab runtime itself**, which needs no local git installation at all:

```
!git clone https://github.com/<org>/<repo-name>.git
%cd <repo-name>/notebooks
```

Cloning inside the Colab runtime is often the more convenient option for the Module 4 and Module 5 notebooks specifically, since it gives them direct access to the real `/data` files via the relative path they already check for first, rather than falling back to the embedded copy.

> **Note:** Colab runtimes are temporary. Anything cloned or uploaded into a runtime disappears when that runtime disconnects or recycles (see below) — re-run your clone/upload step each time you start a fresh session, or save your work to Google Drive.

---

## Inside the Notebook: What to Expect

- **First-run install delay:** the first code cell in every notebook runs a `pip install` for `faiss-cpu`, `numpy`, and `matplotlib`. This typically takes 20–60 seconds the first time you run it in a fresh Colab runtime. Subsequent cells run immediately.
- **Run cells top to bottom, in order.** Several notebooks build up shared state (a corpus, an index, a pipeline object) across cells — skipping ahead will produce `NameError`s, not a sign anything is broken.
- **CPU runtime is sufficient and expected.** If a notebook feels slow, double-check under Runtime → Change runtime type that the hardware accelerator is set to None — a GPU runtime provides no benefit here and only costs you your limited free GPU quota for other work.
- **Runtime disconnects and resets are normal, not errors.** Colab's free tier disconnects idle runtimes after a period of inactivity and periodically recycles active ones. If you reconnect and see `NameError` or missing variables, that's a fresh runtime — just re-run all cells from the top (Runtime → Run all).
- **Saving your work:** Colab does not auto-save to the original GitHub source. Use **File → Save a copy in Drive** early in each session if you want to keep your edits, or **File → Download → Download .ipynb** to save a local copy.
