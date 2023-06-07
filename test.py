dataDir = '../../VQA'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))

annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
quesFile    ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)