# 232
class MyQueue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
        self.front = None

    def push(self, x: int) -> None:
        if not self.stack1: 
            self.front = x
        self.stack1.append(x)


    def pop(self) -> int:
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            self.front = None
        return self.stack2.pop()

    def peek(self) -> int:
        if self.stack2:
            return self.stack2[-1]
        return self.front

    def empty(self) -> bool:
        if not self.stack1 and not self.stack2:
            return True
        return False


# 225
# 155
# 20
# 739
# 503
