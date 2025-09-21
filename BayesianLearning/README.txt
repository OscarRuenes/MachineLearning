Main learns a Naive Bayes learning algorithm assuming features and classification are binary one dimenstional values and no instance has any missing features.
Program should be able to handle any binary classification task with any number of binary-valued attributes.
Program should allow exactly two arguments to be specified in the command line invocation of your program: a training file and a test file.
  In these files, only lines containing non-space characters are relevant. The first relevant line holds the attribute names. Each following relevant line defines a single example, with each column 
  holding this example’s value for the attribute/classification named at the head of the column.
Empty lines may appear anywhere in an input file

To run the program, execute in terminal:
python main.py <path_to_train_file> <path_to_test_file>
