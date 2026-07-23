```bash
# 1.Install it once on Ubuntu (WSL):
sudo apt install git-lfs
# 2.verify
git lfs version
# 3.Initialize Git LFS
git lfs install
# 4.This command:
- configures Git to use LFS on your machine
- updates your Git configuration (~/.gitconfig)
- only needs to be run once per machine/user
- You do not run this inside the virtual environment.
# 5.move to git project and tell Git LFS which file types to manage
git lfs track "*.pkl"
git lfs track "*.parquet"
git lfs track "*.csv"

# 6.This creates a file named : .gitattributes
*.pkl filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.csv filter=lfs diff=lfs merge=lfs -text

# 7. Your Computer
feature_engineered.parquet
        │
        ▼
git add
        │
        ▼
Git creates pointer
        │
        ▼
git commit
        │
        ▼
git push
       / \
      /   \
 Git Repo  Git LFS Storage
(pointer)   (real parquet)

# 8.Suppose i have .csv file pushed on github 10 days back and today i configured git lfs will it move previous pushed csv files on lfs as well ? 

No. Git LFS is not retroactive. It only affects files that are added or recommitted after you've configured it.

```

```bash
# 1. dvc setup
                GitHub
                  │
        ---------------------
        │                   │
   Source Code        maintenance.csv.dvc
                          │
                          ▼
                 DVC Remote Storage
                 ├── AWS S3
                 ├── Azure Blob
                 ├── Google Drive
                 ├── MinIO
                 ├── NAS
                 └── Local Server
# 2.now i have pushed csv files on git lfs when i configured the dvc will it effect ? or it will store both the places ?

Git LFS and DVC are completely independent systems. If you use both on the same file without changing anything, you can end up storing the file in both places, which is usually not what you want.

# 3. Stop tracking CSVs with Git LFS
git lfs untrack "*.csv"

# 4. Commit the updated .gitattributes
# 5. Add the dataset to DVC
ex: dvc add data/raw/maintenance.csv
# 6. commit
ex:
git add data/raw/maintenance.csv.dvc .gitignore
git commit -m "Move dataset tracking from Git LFS to DVC"
# 7. Configure a DVC remote and push
dvc remote add storage <remote>
dvc push
```