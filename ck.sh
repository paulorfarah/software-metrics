#/bin/bash

check_parameter () {
            : ${1?Repo url required.}
    }
check_parameter $1
REPO=$1
PROJECT_NAME=${REPO##*/}
git clone "$REPO" "$PROJECT_NAME"

#ck 
java -jar ck-0.7.1-SNAPSHOT-jar-with-dependencies.jar "${PROJECT_NAME}" true 0 false ./

mv class.csv class_"${PROJECT_NAME}".csv
mv method.csv method_"${PROJECT_NAME}".csv

