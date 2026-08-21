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
# 1. Initialize DVC
dvc init
# 2. Commit
git add .dvc .dvcignore
git commit -m "Initialize DVC"
# 3. Install Google Drive support
uv add "dvc[gdrive]"
# 4. Create Google Drive folder
MLOPS_DVC_STORAGE -- this one you can ignore,as it will be shared with everyone , for onedrive need to create in your system
# 5. Copy folder ID
https://drive.google.com/drive/folders/1LeyCC1gNR8MTqhleVfWMobOWWvVgOqod
# 6. Configure DVC Remote
dvc remote add -d gdrive gdrive://1LeyCC1gNR8MTqhleVfWMobOWWvVgOqod # this for google drive , doesn't work due to google block the local url,
dvc remote add -d onedrive_local /mnt/c/Users/<your_username>/OneDrive/MLOPS_DVC_STORAGE # this works fine 
ex : dvc remote add -d onedrive /mnt/c/Users/admin/OneDrive/MLOPS_DVC_STORAGE
# 7. Verify
dvc remote list
# 8. Commit remote configuration
git add .dvc/config
git commit -m "Configure DVC remote"
# 9. Authenticate
dvc push
# 10. Add the dataset to DVC (Track Dataset)
ex: dvc add data/raw/maintenance.csv
# 11. commit
ex:
git add data/raw/maintenance.csv.dvc .gitignore
git commit -m "Track raw dataset"
# 12. push the code
git push
# 13. push the data
dvc push

# 1. dvc and github

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
```

```bash
# normal flow goes
dvc add data/raw/maintenance.csv
git add .
git commit -m "Track dataset"
git push          # Push code to GitHub
dvc push          # Upload data to Google Drive
```

```bash
# operational useful
ex:
# 1.Stop tracking with DVC and remove its cache:
dvc remove data/raw/maintenance.csv.dvc
# 2. Delete the files physically:
rm data/raw/maintenance.csv data/raw/maintenance.csv.dvc
# 3. Stage the removal in Git:
git add data/raw/.gitignore
git rm data/raw/maintenance.csv.dvc
# 4. Track the new file with DVC:
dvc add data/raw/ai4i2020.csv
# 5. Commit the changes to Git
git add data/raw/ai4i2020.csv.dvc data/raw/.gitignore
git commit -m "ai4i2020.csv dataset"
# 6. Push your code changes to GitHub:
git push 
# 7. Push the new dataset to your DVC remote storage:
dvc push
```