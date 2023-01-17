#/bin/bash

check_parameter () {
	    : ${1?Repo url required.}
    }
check_parameter $1

# set target directories
DIR_A="projectA/"
DIR_B="projectB/"

# clean target directories
rm -Rf "$DIR_A"
rm -Rf "$DIR_B"

#clone repository to target directories
REPO=$1
PROJECT_NAME=${REPO##*/}
DIR_A="${DIR_A}${PROJECT_NAME}"
DIR_B="${DIR_B}${PROJECT_NAME}"

git clone "$REPO" "$DIR_A"
git clone "$REPO" "$DIR_B"
#echo "Script executed from: ${PWD}"
rm -Rf nohup.out
nohup python3 refactoring.py --git_repo_folder_A "${DIR_A}" --git_repo_folder_B "${DIR_B}" --project_name "${PROJECT_NAME}" --commits "jgit.txt" --mode "csv" &
#echo "nohup python3 evometrics.py --pathA ${DIR_A} --pathB ${DIR_B} --commits ${PROJECT_NAME}.txt --project_name ${PROJECT_NAME} --absolute_path ${PWD}/ --mode \"csv\" &"
