with open("ns_tweets.csv") as readf:
    with open("two_labels.csv", "w") as writef:
        for index, line in enumerate(readf):
            if line[0] != "1":
                writef.write(line)