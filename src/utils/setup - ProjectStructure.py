import os

# Get the project root:
# create_structure.py -> utility -> src -> project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Define project files relative to the project root
files = [
    # Root files
    "README.md",
    ".gitignore",
    ".dvcignore",
    ".gitattributes",
    "pyproject.toml",

    # GitHub Actions
    ".github/workflows/ci.yaml",


    # Configuration
    "configs",

    # Airflow
    "airflow/dags/airflow_dag.py",
    "airflow/dags/common_dag.py",

    # folders
    "app",
    "data",
    "k8s",
    "mlflow",
    "monitoring",
    "notebooks",
    "outputs",
    "tests",
    "src",


    # Docker
    "docker/docker-compose.yml",


    # Existing requirement files
    "requirements_full_wip.txt",
    "requirements-dev.txt",
]

# Create directories and files relative to project root
for file_path in files:

    # Convert relative path into absolute path
    full_path = os.path.join(PROJECT_ROOT, file_path)

    # Get directory containing the file
    directory = os.path.dirname(full_path)

    # Create directory if it doesn't exist
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Create file only if it doesn't already exist
    if not os.path.exists(full_path):
        with open(full_path, "w") as f:
            pass

        print(f"Created: {file_path}")

    else:
        print(f"Skipped: {file_path} already exists")

print("\nProject structure updated successfully!")