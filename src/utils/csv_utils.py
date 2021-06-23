import csv


def write_csv(csv_file, csv_content, csv_headers):
    """
    Write the data to a csv file.

    Parameters
    ----------
    csv_file : str
        Output path of csv file
    csv_content : 2D list
        A nested list including the row content for the csv file
    csv_headers : list
        Column names
    """

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
        writer.writerows(csv_content)
