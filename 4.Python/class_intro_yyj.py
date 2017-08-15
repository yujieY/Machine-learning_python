#  -*- coding:utf-8 -*-


class People:
	def __init__(self,n,a,s):
		self.name = n
		self.age = a
		self.__score = s
		self.print_people()


	def print_people(self):
		str = u'%s的年龄：%d，成绩为：%.2f' % (self.name,self.age,self.__score)
		print str

class Student(People):
	def __init__(self,n,a,w):
		People.__init__(self,n, a, w)
		self.name = 'student' + self.name

	def print_people(self):
		str = u'%s的年龄：%d' % (self.name,self.age)
		print str

def func(p):
	p.age = 11


if __name__ == '__main__':
	p = People('Tom', 10, 3.14159)
	# func(p)
	# p.print_people()
	# print

	j = Student('gerry', 12, 2.71828)
	# print
	# print j.print_people()

	People.print_people(p)
	Student.print_people(j)
