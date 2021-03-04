import numpy as np

# Demo whitespace, range, printing
# Prints 0-9 ten times, declaring itself done each time it finishes
def demo1():
    for i in range(10):    # range(10) = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for j in range(10):
            print(j)
        print('set ' + str(i) + ' done') # deindent puts this outside the 'for' 

# Demo dictionary, multiple return values, foreach, len
def demo2():
    # Initialize a lookup dictionary with curly braces; could be empty {}
    mycolors = {
        "white" : 255,
        "gray" : 128,
        "black" : 0
    }
    mycolors["gray"] = 127
    for colorword in mycolors:
        print(colorword + " is " + str(mycolors[colorword]))
    # Functions can return multiple values
    return len(mycolors), mycolors["gray"]  # len works for lists, too
    
# Find a dot product of an input vector with the range vector of same length
# Also demonstrate "elif/else"
def demo3(input):
    if len(input) < 1:
        print("No degenerate input please")
    elif len(input) > 10:
        print("Bad demo - nobody will mentally compute that number")
    else:
        return np.dot(input,range(len(input)))
    return np.nan  # "Not a number"

# Show how operations often apply to whole lists
def demo4(mylist):
	return 3 * np.sin(np.array(mylist))

# Class example - just a container with a number
class DemoClass:
    def __init__(self, magic_number):   # constructor must be __init__
        self.mynum = magic_number

    def magic(self):   # self, the object, is always an argument to these methods
        return self.mynum

    def mysquare(self):
        return self.mynum*self.mynum

# A thing that lets us know we ran this file like a program
print("Hello, world!")
        
