import csv
import json


def parse_file():
    rows = []
    with open("SOC_remformatted.csv") as file:
        lines = csv.reader(file)
        next(lines)
        for line in lines:
            # Drop *_code from the data
            line_content = [line[2], line[4], line[6], line[8]]
            rows.append('|'.join(line_content))

    return rows


def write_json():
    rows = parse_file()
    with open("soc_full_line.json", "w") as file:
        file.write(json.dumps(rows))



def main():
    write_json()


if __name__ == '__main__':
    main()
