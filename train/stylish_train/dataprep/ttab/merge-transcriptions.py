import sys, argparse


def parse(f):
    result = {}
    chapter = ""
    for line in f:
        fields = line.split("|")
        if fields[0] == "chapter":
            chapter = fields[1].strip()
            result[chapter] = {}
        else:
            phrase = ""
            if fields[0] == "phrase":
                phrase = fields[3].strip()
            name = "%08d|%08d" % (int(fields[1].strip()), int(fields[2].strip()))
            result[chapter][name] = phrase
    return result


def merge(left, right):
    for chapter in left.keys():
        for key in right[chapter].keys():
            if key not in left[chapter] or left[chapter][key] == "":
                left[chapter][key] = right[chapter][key]


def printout(chapter_list):
    for chapter in chapter_list.keys():
        print("chapter|" + chapter)
        for time_range in sorted(chapter_list[chapter].keys()):
            segment = chapter_list[chapter][time_range]
            if segment == "":
                print("skipped|" + split_time(time_range))
            else:
                print("phrase|" + split_time(time_range) + "|" + segment)


def split_time(time):
    fields = time.split("|")
    return str(int(fields[0])) + "|" + str(int(fields[1]))


parser = argparse.ArgumentParser()
parser.add_argument("left")
parser.add_argument("right")
args = parser.parse_args()

left_chapters = {}
right_chapters = {}
with open(args.left, "r") as l:
    left_chapters = parse(l)
with open(args.right, "r") as r:
    right_chapters = parse(r)
merge(left_chapters, right_chapters)
printout(left_chapters)
