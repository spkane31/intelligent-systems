import csv
test = "0123456789."

data = []

def processData():
    count = 0
    stressed = True
    with open("rawdata.txt", "r") as fp:
        line = fp.readline()
        
        while line:
            line = fp.readline()
            if count > 200:
                stressed = False
            boole = False
            w = ""
            h = ""
            for t in line:
                if t in "0123456789.":
                    if not boole:
                        w += t
                    else:
                        h += t
                else:
                    boole = True
            try: 
                if stressed:
                    data.append([float(w),float(h), '1'])
                    count += 1
                else:
                    data.append([float(w),float(h), '0'])
                    count += 1
                    
            except:
                "Nothing"
    # print(data)
# print(data)
    f = open("demo.txt", "a")
    f.seek(0)
    f.truncate()
    # f.write("ANOTHA")
    for d in data:
        string = str(d[0]) + ", " + str(d[1]) + ", '" + str(d[2]) + "'\n"
        f.write(string)