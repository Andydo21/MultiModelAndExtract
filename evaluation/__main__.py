import argparse

from .cli import add_arguments, dispatch


def main():
    parser=argparse.ArgumentParser(description='Independent document QA evaluation; no synthetic gold or results')
    add_arguments(parser)
    options=vars(parser.parse_args())
    try:
        print(dispatch(options))
    except ValueError as error:
        parser.error(str(error))


if __name__=='__main__':
    main()
