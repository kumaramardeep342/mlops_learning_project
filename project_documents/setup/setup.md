# SETUP.md

# Environment Setup Guide

This document explains how to set up the local development environment for the MLOps Project.

---

# Prerequisites

* Windows 11
* WSL2
* Ubuntu 26.04 LTS
* GitHub Account
* Docker Desktop
* VS Code

---

# 1. Setup WSL2

Verify WSL installation:

```bash
wsl -l -v
```

Verify Ubuntu version:

```bash
lsb_release -a
```

Expected:

```text
Ubuntu 26.04 LTS
```

---

# 2. Install Git

Verify Git installation:

```bash
git --version
```

If not installed:

```bash
sudo apt update
sudo apt install git -y
```

Configure Git:

```bash
git config --global user.name "<github-username>"
git config --global user.email "<github-email>"
```

Verify:

```bash
git config --global --list
```

---

# 3. Install Python 3.14 and UV
- 3.14 will be globally, so once create virtual env install 3.12 inside virtual environment

Verify Python:

```bash
python3 --version
```

Expected:

```text
Python 3.14.x
```

Install UV:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload shell:

```bash
source ~/.bashrc
```

Verify:

```bash
uv --version
```

---

# 4. Create Virtual Environment

Navigate to project folder:

```bash
cd ~/projects/mlops_project
```

Create virtual environment:

```bash
uv venv
```

Activate:

```bash
source .venv/bin/activate
```

Verify:

```bash
which python
python --version
```

---

# 5. Install Java 17

Install OpenJDK:

```bash
sudo apt update
sudo apt install openjdk-17-jdk -y
```

Verify:

```bash
java -version
javac -version
```

---

# 6. Install Apache Spark 4.1.2

Create applications directory:

```bash
mkdir -p ~/apps
cd ~/apps
```

Download Spark:

```bash
wget https://downloads.apache.org/spark/spark-4.1.2/spark-4.1.2-bin-hadoop3.tgz
```

Extract:

```bash
tar -xzf spark-4.1.2-bin-hadoop3.tgz
```

Rename:

```bash
mv spark-4.1.2-bin-hadoop3 spark
```

---

### 6.1 Configure Environment Variables

Edit bash profile:

```bash
nano ~/.bashrc
```

Add:

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

export SPARK_HOME=$HOME/apps/spark

export PATH=$JAVA_HOME/bin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
```

Reload:

```bash
source ~/.bashrc
```

---

### 6.2 Verify Spark Installation

Check Spark home:

```bash
echo $SPARK_HOME
```

Verify Spark:

```bash
spark-submit --version
```

Expected:

```text
Spark 4.1.2
Java 17
```

---

# 9. Install VS Code

Install VS Code from Microsoft Store.

Enable below once wsl enabled in vs code other it will installed locally:

* WSL Integration
* Python Extension
* Pylance Extension
* Jupyter Extension
* SQLite Extension

Open project from Ubuntu:

```bash
cd ~/projects/mlops_project
code .
```

Verify VS Code shows:

```text
WSL: Ubuntu
```

---

# 10. Install Docker Desktop

Install Docker Desktop from Microsoft Store.

Enable WSL Integration:

Docker Desktop → Settings → Resources → WSL Integration

Enable:

```text
Ubuntu
```

Apply and Restart.

---

### 10.1 Verify Docker

From Ubuntu terminal:

```bash
docker --version
docker compose version
docker run hello-world
```

Expected:

```text
Hello from Docker!
```

---

# 12. Project Initialization

Clone repository:

```bash
git clone <repository-url>
```

Navigate:

```bash
cd mlops_project
```

Activate environment:

```bash
source .venv/bin/activate
```

---

# Verification Checklist

* [ ] WSL2 Installed
* [ ] Ubuntu 26.04 Running
* [ ] Git Configured
* [ ] Python 3.14 Installed
* [ ] UV Installed
* [ ] Virtual Environment Created
* [ ] Java 17 Installed
* [ ] Spark 4.1.2 Installed
* [ ] VS Code Connected to WSL
* [ ] Docker Desktop Installed
* [ ] Docker WSL Integration Enabled
* [ ] Repository Cloned Successfully

Environment setup complete.
