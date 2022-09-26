#/bin/bash

check_parameter () {
            : ${1?Repo url required.}
    }
check_parameter $1

REPO=$1
PROJECT_NAME=${REPO##*/}
filename="${PROJECT_NAME}.txt"
mkdir -p organic-results/"${PROJECT_NAME}"
mkdir -p "${PROJECT_NAME}"
git clone "$REPO" "$PROJECT_NAME"

while read commit; do
# reading each commit 
echo "$commit"
echo "$PWD"

cd "${PROJECT_NAME}" 
git checkout $commit
cd ..
/bin/sh organic-v0.1.0/bin/organic -sf organic-results/"${PROJECT_NAME}"/"${PROJECT_NAME}"_"${commit}"_organic.json -src "$PWD"/"${PROJECT_NAME}" 
done < $filename
rm -Rf "${PROJECT_NAME}"
