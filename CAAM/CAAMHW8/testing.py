def main():
    index = 0
    state = "initial"
    while(True):
        
        #Infinite States.
        if state == "initial":
            print("Setting up...")
            hello = input("Hello")
            state = "dictionary"
        elif state == "dictionary":
            print("Hello. You are in the dictionary state.")
            state = dictionary()
        elif state == "abnormal":
            print("Hello. You are in the abornmal state.")
            break
    #print("Exited with code 0")
        
def dictionary():
    desc = ["desc1", "desc2", "desc3"]
    while(True):
        print(desc, "Please Select one of the above, via 1, 2, 3")
        command = input("YAHR")
        if command == "1":
            print(desc[0])
            return "initial"
        elif command == "2":
            print(desc[1])
            return "dictionary"
        elif command == "3":
            print(desc[2])
            return "abnormal"
        
main()





    
