# GitHub_Actions

> **AI generation note:** This repository's README and supporting GitHub Actions setup were generated with assistance from OpenAI ChatGPT and may be reviewed or modified manually over time.

Central public repository for GitHub Actions that support projects whose main repositories can remain separate or private.

This repository is designed to be reusable. Each project can have one or more workflows under `.github/workflows/`, while any scripts, frozen datasets, snapshots, configuration files, or other files required by those workflows live in a project-specific folder at the repository root.

## Repository Structure

```text
GitHub_Actions/
├── .github/
│   └── workflows/
│       ├── keep-awake-pulseguard.yml
│       ├── determinism-pulseguard.yml
│       ├── keep-awake-hybrid-option-pricer.yml
│       ├── determinism-hybrid-option-pricer.yml
│       ├── keep-awake-fraud-detection.yml
│       ├── determinism-fraud-detection.yml
│       └── ...future workflows...
│
├── PulseGuard_Risk_Monitor_Model/
│   └── files required by its determinism checker
│
├── Hybrid_Option_Pricer/
│   └── files required by its determinism checker
│
├── Graph-Based_Financial_Fraud_Detection/
│   └── files required by its determinism checker
│
├── ...future project folders...
└── README.md
```

## Current Projects and Workflows

### PulseGuard_Risk_Monitor_Model

Hosted workflows:

- **Keep Streamlit Awake (PulseGuard_Risk_Monitor_Model)**
  - Runs on a recurring schedule.
  - Visits the deployed Streamlit application to help prevent it from going idle.
  - Does not require the full private project repository.

- **Determinism Checker (PulseGuard_Risk_Monitor_Model)**
  - Runs once per day and can also be triggered manually.
  - Uses the files stored under `PulseGuard_Risk_Monitor_Model/`.
  - Uses frozen inputs/snapshots and the minimal project logic required to verify deterministic output.

### Hybrid_Option_Pricer

Hosted workflows:

- **Keep Streamlit Awake (Hybrid_Option_Pricer)**
  - Runs on a recurring schedule.
  - Visits the deployed Streamlit application to help prevent it from going idle.

- **Determinism Checker (Hybrid_Option_Pricer)**
  - Runs once per day and can also be triggered manually.
  - Uses the files stored under `Hybrid_Option_Pricer/`.
  - Uses the frozen market-data snapshot and minimal model/checking code required for the determinism test.

### Graph-Based_Financial_Fraud_Detection

Hosted workflows:

- **Keep Streamlit Awake (Graph-Based_Financial_Fraud_Detection)**
  - Runs on a recurring schedule.
  - Visits the deployed Streamlit application to help prevent it from going idle.

- **Determinism Checker (Graph-Based_Financial_Fraud_Detection)**
  - Runs once per day and can also be triggered manually.
  - Uses the files stored under `Graph-Based_Financial_Fraud_Detection/`.
  - Generates deterministic synthetic data as part of the check, so it does not require a frozen project-data snapshot.

## Adding a New Project in the Future

A new project does **not** need its own GitHub Actions repository. Add its Actions here instead.

### 1. Create a project folder if the workflows need supporting files

At the repository root, create a folder matching the main project name, for example:

```text
New_Project/
```

Only add files that the Actions actually require, such as:

- determinism-check scripts
- test scripts
- frozen input datasets
- expected-output snapshots
- requirements files
- small configuration files

Do not copy the entire project into this repository unless a workflow genuinely requires it.

If a workflow is completely self-contained, no project folder is necessary.

### 2. Add the workflow YAML file

All GitHub Actions workflow definitions belong in:

```text
.github/workflows/
```

Use a descriptive filename, for example:

```text
keep-awake-new-project.yml
determinism-new-project.yml
nightly-tests-new-project.yml
```

### 3. Follow the workflow naming convention

The displayed workflow name should use:

```text
Workflow Name (Main_Project_Name)
```

Examples:

```text
Keep Streamlit Awake (New_Project)
Determinism Checker (New_Project)
Nightly Tests (New_Project)
Dependency Check (New_Project)
```

This makes it immediately clear in the GitHub Actions interface which main project each workflow belongs to.

### 4. Point the workflow at its project folder

If the workflow needs local supporting files, make sure commands use the relevant directory. For example:

```yaml
- name: Run determinism check
  working-directory: New_Project
  run: python determinism_check.py
```

### 5. Update this README

Add the new project under **Current Projects and Workflows** and briefly document:

- the workflow names
- what they do
- their schedule or trigger
- which local files/snapshots they require

## Workflow Design Guidelines

When adding future Actions:

- Keep each workflow associated with a clearly named main project.
- Store workflow definitions only in `.github/workflows/`.
- Keep supporting files isolated inside that project's root folder.
- Prefer minimal copies of project code rather than duplicating an entire private repository.
- Never commit API keys, access tokens, passwords, private keys, or other credentials.
- Use GitHub Actions Secrets for any credentials a workflow requires.
- Prefer `workflow_dispatch` in addition to scheduled triggers when manual testing would be useful.
- Avoid unnecessary workflow triggers such as `push` or `pull_request` if a job only needs to run on a schedule.
- Keep filenames and displayed workflow names consistent and descriptive.

## Public Repository Note

This repository is intended to be public. Therefore, every file committed here should be safe for public disclosure.

Do **not** place proprietary source code, private datasets, credentials, secrets, or other sensitive information in this repository. If an Action needs confidential information, store it using GitHub Actions Secrets or another appropriate secure mechanism instead of committing it.

The main project repositories can remain private while this repository contains only the minimal public files needed to run their external Actions.
