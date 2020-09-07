from pathlib import Path
import os
import os.path


class settingReader():
    
    def __init__(self):
        
        mod_path = Path(__file__).parent
        f = open((mod_path / "settings.txt").resolve(), "r")
        dicts = {}
        for x in f:
            if x[0] == "~":
                sliceObject = slice(3,-1)
                splitString = x[sliceObject].split("=")
                if x[1] == "S":
                    dicts[splitString[0].strip()] = splitString[1].strip()
                elif x[1] == "T":
                    tempList = []
                    for entry in splitString[1].split(","):
                        tempList.append(entry.strip())
                    dicts[splitString[0].strip()] = tuple(tempList)
                elif x[1] == "B":
                    if splitString[1].strip() == "True":
                        dicts[splitString[0].strip()] = True
                    else:
                        dicts[splitString[0].strip()] = False
                elif x[1] == "I":
                    dicts[splitString[0].strip()] = int(splitString[1].strip())
                
        f.close()
        self.variableValues = dicts
    def getItem(self,variableName):
        return self.variableValues[variableName]
