def line2offset(f):
    offset = 0
    lines = []

    for line in f:
        lines.append(offset)
        offset += len(line)

    return lines


def seek1line(f, lines, line_num):
    f.seek(lines[line_num])
    line = f.readline()
    return line.split()
