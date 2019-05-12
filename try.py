with open("ns_tweets.csv") as readf:
    for index, line in enumerate(readf):
        if line[0] != "0" and line[0] != "1" and line[0] != "2":
            print(line)
            print(index)