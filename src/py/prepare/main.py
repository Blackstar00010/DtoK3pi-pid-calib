import dataprep, plot_dists
import src.py.utils.utils as utils

@utils.alert
def main():
    dataprep.check_proced_mc(True, True)
    plot_dists.main()


if __name__ == '__main__':
    main()
