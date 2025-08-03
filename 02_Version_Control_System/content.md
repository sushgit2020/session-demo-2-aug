# Version Control System - Level 100 Theory

## 1. Git Essentials

Git is a distributed version control system that tracks changes in source code during software development. In MLOps, Git manages not just code but also configurations, documentation, and pipeline definitions.

### Core Git Concepts

#### Repository (Repo)
A Git repository contains:
- **Working Directory**: Current files you're editing
- **Staging Area**: Files prepared for commit
- **Git Directory**: Metadata and object database

#### Commits
- **Atomic Changes**: Each commit represents a single logical change
- **Commit Hash**: Unique SHA-1 identifier (e.g., `a1b2c3d4`)
- **Commit Message**: Descriptive text explaining the change

#### Branches
- **Master/Main**: Primary development branch
- **Feature Branches**: Isolated development for new features
- **Release Branches**: Prepare for production releases
- **Hotfix Branches**: Emergency fixes for production

### Git Object Model
```
Commit -> Tree -> Blob
   |        |      |
   |        |      └── File content
   |        └── Directory structure
   └── Metadata (author, timestamp, parent)
```

## 2. Configuring Git

### Global Configuration
```bash
# User identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Default editor
git config --global core.editor "code --wait"

# Line ending handling
git config --global core.autocrlf true  # Windows
git config --global core.autocrlf input # macOS/Linux
```

### Repository-Specific Configuration
```bash
# Different email for work projects
git config user.email "work.email@company.com"

# Custom merge tool
git config merge.tool vimdiff
```

### SSH Key Setup
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to SSH agent
ssh-add ~/.ssh/id_ed25519

# Test connection
ssh -T git@github.com
```

## 3. Branching

### Branching Strategies

#### Git Flow
- **develop**: Integration branch for features
- **feature/***: New feature development
- **release/***: Release preparation
- **hotfix/***: Emergency fixes
- **master or main**: Production-ready code

#### GitHub Flow
- **master or main**: Production branch
- **feature-branches**: Short-lived feature development
- **Pull Requests**: Code review and integration

#### GitLab Flow
- **master**: Development branch
- **production**: Deployment branch
- **environment branches**: Staging, testing environments

### Branch Operations
```bash
# Create and switch to branch
git checkout -b feature/user-authentication

# Switch between branches
git checkout main
git checkout feature/user-authentication

# Merge branches
git checkout main
git merge feature/user-authentication

# Delete branch
git branch -d feature/user-authentication
```



## 4. Git Workflow

### Feature Development Workflow
1. **Create Feature Branch**
   ```bash
   git checkout -b feature/new-model
   ```

2. **Develop and Commit**
   ```bash
   git add .
   git commit -m "Add initial model structure"
   ```

3. **Push to Remote**
   ```bash
   git push -u origin feature/new-model
   ```

4. **Create Pull Request**
   - Code review process
   - Automated testing
   - Discussion and feedback

5. **Merge to Main**
   ```bash
   git checkout main
   git pull origin main
   git merge feature/new-model
   ```

### Collaborative Workflow
1. **Fork Repository** (for external contributors)
2. **Clone Locally**
   ```bash
   git clone https://github.com/username/repo.git
   ```

3. **Keep Fork Updated**
   ```bash
   git remote add upstream https://github.com/original/repo.git
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

### Merge vs. Rebase

#### Merge
- Preserves commit history
- Creates merge commits
- Better for collaborative workflows

#### Rebase
- Linear commit history
- Cleaner project history
- Good for feature branches

```bash
# Merge workflow
git checkout main
git merge feature-branch

# Rebase workflow
git checkout feature-branch
git rebase main
git checkout main
git merge feature-branch
```

## 5. Repository (Repo)

### Repository Types

#### Local Repository
- Complete Git history on your machine
- Full functionality offline
- Local branches and commits

#### Remote Repository
- Hosted on platforms (GitHub, GitLab, Bitbucket)
- Collaboration hub
- Backup and synchronization

#### Bare Repository
- No working directory
- Used for shared repositories
- Common in CI/CD pipelines

### Repository Structure
```
.git/
├── HEAD              # Current branch pointer
├── config            # Repository configuration
├── objects/          # Git objects (commits, trees, blobs)
├── refs/             # Branch and tag references
├── hooks/            # Git hooks
└── index             # Staging area
```

### Remote Management
```bash
# View remotes
git remote -v

# Add remote
git remote add origin https://github.com/user/repo.git

# Change remote URL
git remote set-url origin git@github.com:user/repo.git

# Remove remote
git remote remove origin
```

## 6. Git Commands

### Basic Commands
```bash
# Initialize repository
git init

# Clone repository
git clone <url>

# Check status
git status

# View differences
git diff
git diff --staged

# Add files
git add file.txt
git add .              # Add all files
git add *.py           # Add Python files

# Commit changes
git commit -m "Commit message"
git commit -am "Add and commit"

# View history
git log
git log --oneline
git log --graph
```

### Advanced Commands
```bash
# Interactive staging
git add -p

# Amend last commit
git commit --amend

# Reset changes
git reset HEAD~1       # Undo last commit
git reset --hard HEAD  # Discard all changes

# Stash changes
git stash
git stash pop
git stash list

# Cherry-pick commits
git cherry-pick <commit-hash>

# Revert commits
git revert <commit-hash>
```

### Inspection Commands
```bash
# Show commit details
git show <commit-hash>

# Find changes in history
git grep "search-term"

# Blame/annotate
git blame file.txt

# Find when bug was introduced
git bisect start
git bisect bad <commit>
git bisect good <commit>
```

## 7. GitHub Actions

GitHub Actions is a CI/CD platform that automates workflows directly in your GitHub repository.

### Core Concepts

#### Workflows
- YAML files in `.github/workflows/`
- Triggered by events (push, pull request, schedule)
- Contains one or more jobs

#### Jobs
- Run on virtual machines (runners)
- Contain steps that execute actions
- Can run in parallel or sequence

#### Actions
- Reusable units of code
- Can be custom or from marketplace
- Perform specific tasks

### Basic Workflow Structure
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/
```

### MLOps-Specific Workflows

#### Model Training Pipeline
```yaml
name: Model Training

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  train:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Download data
      run: python scripts/download_data.py
    
    - name: Train model
      run: python scripts/train_model.py
    
    - name: Evaluate model
      run: python scripts/evaluate_model.py
    
    - name: Upload model artifacts
      uses: actions/upload-artifact@v3
      with:
        name: model-artifacts
        path: models/
```

#### Model Deployment Pipeline
```yaml
name: Deploy Model

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t ml-model:${{ github.sha }} .
    
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
    
    - name: Run smoke tests
      run: |
        python tests/smoke_tests.py
    
    - name: Deploy to production
      if: success()
      run: |
        echo "Deploying to production environment"
```

### Advanced Features

#### Matrix Builds
```yaml
strategy:
  matrix:
    python-version: [3.8, 3.9, '3.10']
    os: [ubuntu-latest, windows-latest, macOS-latest]

runs-on: ${{ matrix.os }}

steps:
- name: Set up Python ${{ matrix.python-version }}
  uses: actions/setup-python@v3
  with:
    python-version: ${{ matrix.python-version }}
```

#### Secrets Management
```yaml
steps:
- name: Deploy to AWS
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
  run: |
    aws s3 cp model.pkl s3://my-model-bucket/
```

#### Conditional Execution
```yaml
steps:
- name: Deploy only on main branch
  if: github.ref == 'refs/heads/main'
  run: |
    echo "Deploying to production"

- name: Run integration tests
  if: contains(github.event.head_commit.message, '[integration]')
  run: |
    pytest tests/integration/
```

### Best Practices for MLOps

#### 1. Environment Isolation
```yaml
- name: Create virtual environment
  run: |
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
```

#### 2. Caching Dependencies
```yaml
- name: Cache pip dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
```

#### 3. Artifact Management
```yaml
- name: Save model artifacts
  uses: actions/upload-artifact@v3
  with:
    name: trained-model-${{ github.sha }}
    path: |
      models/
      metrics/
      logs/
```

#### 4. Notification and Reporting
```yaml
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    text: 'Model training failed!'
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### GitHub Actions for ML Workflows

#### Data Validation
```yaml
name: Data Validation

on:
  push:
    paths:
      - 'data/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Validate data schema
      run: python scripts/validate_data.py
    - name: Check data quality
      run: python scripts/data_quality_checks.py
```

#### Model Performance Monitoring
```yaml
name: Performance Monitor

on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours

jobs:
  monitor:
    runs-on: ubuntu-latest
    steps:
    - name: Check model performance
      run: python scripts/performance_monitor.py
    - name: Alert on degradation
      if: failure()
      run: python scripts/send_alert.py
```

This comprehensive coverage of version control systems provides the foundation for managing ML projects, code collaboration, and automated workflows essential for MLOps success.