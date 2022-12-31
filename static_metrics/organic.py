import json
import pydriller
import csv
import git

row = ["projectName", "commitNumber", "fullyQualifiedName",
       "PublicFieldCount", "IsAbstract", "ClassLinesOfCode", "WeighOfClass",
       "FANIN", "TightClassCohesion", "FANOUT", "OverrideRatio", "LCOM3",
       "WeightedMethodCount", "LCOM2", "NumberOfAccessorMethods",
       'LazyClass', 'DataClass', 'ComplexClass', 'SpaghettiCode',
       'SpeculativeGenerality', 'GodClass', 'RefusedBequest',
       'ClassDataShouldBePrivate', 'BrainClass', 'TotalClass',
       'LongParameterList', 'LongMethod', 'FeatureEnvy',
       'DispersedCoupling', 'MessageChain', 'IntensiveCoupling',
       'ShotgunSurgery', 'BrainMethod', 'TotalMethod', 'TotalClassMethod',
       "DiversityTotal", "DiversityMethod", "DiversityClass"]

smells_template = {
        # CLASSE
        "LazyClass": 0,
        "DataClass": 0,
        "ComplexClass": 0,
        "SpaghettiCode": 0,
        "SpeculativeGenerality": 0,
        "GodClass": 0,
        "RefusedBequest": 0,
        "ClassDataShouldBePrivate": 0,
        "BrainClass": 0,
        "TotalClass": 0,
        # METODO
        "LongParameterList": 0,
        "LongMethod": 0,
        "FeatureEnvy": 0,
        "DispersedCoupling": 0,
        "MessageChain": 0,
        "IntensiveCoupling": 0,
        "ShotgunSurgery": 0,
        "BrainMethod": 0,
        "TotalMethod": 0,
        # class + method
        "TotalClassMethod": 0,
        "DiversityTotal": 0,
        "DiversityMethod": 0,
        "DiversityClass": 0
    }

projects = ['commons-bcel', 'commons-csv', 'easymock', 'gson', 'Openfire', 'commons-text', 'commons-io', 'pdfbox', 'dubbo']
for project_name in projects:
    csvPath = 'results/organic/' + project_name
    with open(csvPath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(row)

        organicRepo = pydriller.Git(project_name)
        repo = git.Repo(project_name)
        tags = repo.tags
        for tag in tags:
            hashCurrent = organicRepo.get_commit_from_tag(tag.name).hash
            try:
                # Carrega o arquivo do organic
                # commons-bcel_3aecd517ad0ac4c83828a5f89b6b062acb6f4f6a_organic.json
                path_organic_results = 'organic_results/' + project_name + '/' + project_name + '_' + hashCurrent + "_organic.json"
                file = open(path_organic_results )
                smells_classes = json.load(file)
                for _class in smells_classes:
                    smells = smells_template.copy()
                    for _smell in _class['smells']:
                        smells[_smell['name']] = smells[_smell['name']] + 1
                        smells["TotalClass"] = smells["TotalClass"] + 1
                        if (smells[_smell['name']] == 1):
                            smells["DiversityClass"] = smells["DiversityClass"] + 1

                    for _method in _class['methods']:
                        for _smell in _method['smells']:
                            smells[_smell['name']] = smells[_smell['name']] + 1
                            smells["TotalMethod"] = smells["TotalMethod"] + 1
                            if smells[_smell['name']] == 1:
                                smells["DiversityMethod"] = smells["DiversityMethod"] + 1
                        #   writer.writerow(_smell['name'])
                    # projectName = "commons-bcel"
                    className = _class['fullyQualifiedName']
                    commitVersion = hashCurrent
                    smells["TotalClassMethod"] = smells["TotalMethod"] + smells["TotalClass"]
                    smells["DiversityTotal"] = smells["DiversityMethod"] + smells["DiversityClass"]
                    _row = [project_name, commitVersion, className]
                    metricsValuesArray = _class["metricsValues"]

                    for m in metricsValuesArray:
                        _row.append(metricsValuesArray[m])
                    for s in smells:
                        _row.append(smells[s])

                    writer.writerow(_row)


            except Exception as e:
                print(hashCurrent)
                # print(e)

print('the end: ' + project_name)