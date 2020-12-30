
def test_docker(input_dir, output_dir):
    """
    Quick test to see whether docker is setup correctly.
    """

    print("I am talking to you from inside your container!")

    if input_dir.is_dir():
        print("Input is visible")

    if output_dir.is_dir():
        print("output is visible")

    print("Here are the files in the inputs directory")
    for file in input_dir.iterdir():
        print(file)