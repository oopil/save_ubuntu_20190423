import os, openpyxl
class Printer():
	def __init__(self):
		pass

	def p_highlight(self, line):
		line_ = '-'*50
		print(line_)
		print(line)
		print(line_)

	def p_list(self, l):
		print('print list')
		for line in l:
			print(line)

	def p_list_index(self, l, index):
		print('print list by index split /')
		for line in l:
			name = line.split('/')
			print(name[index])

#%%
class FileReader():
	def __init__(self):
		pass
	
	def open_file(self, file_name):
		self.file_name = file_name
		self.fd = open(self.file_name, 'r+')
	
	def get_all_lines(self):
		return self.fd.readlines()
#%%
class XL():
	def __init__(self, file_name, rd_wr):
		self.file_name = file_name
		self.open_xl(file_name, rd_wr)
		self.ws = []
		self.ws.append(self.cur_ws)

	def get_sh_names(self):
		self.ws_list = self.book.sheetnames
		print(self.ws_list)
		return self.ws_list
	
	def open_xl(self, file_name, read_write):
		print('in file_open, xl file_name is', file_name)
		if read_write == 'read':
			self.book = openpyxl.Workbook()

			# self.book = openpyxl.load_workbook(file_name)
		else :
			self.book = openpyxl.Workbook()
# need get_workload function to read lines
		self.cur_ws = self.book.active

	def open_ws(self, ws_name):
		self.cur_ws = self.book[ws_name]

	def rd_all_row(self):
		rows = []
		for line in self.cur_ws.rows:
			rows.append(self.get_line(line))
		return rows

	def get_line(self, row):
		line = []
		for word in row:
			line.append(word.value)
		return line


	def set_cur_ws(self, index):
		self.cur_ws = self.ws[index]
	
	def write(self, row, col, content):
		 # row, col and content should be strings
		pos = col+row
		 #print(pos, content)
		self.cur_ws[pos] = content

	def save_xl(self):
		self.save_xl_as(self.file_name)

	def save_xl_as(self, file_name):
		self.book.save(file_name)
	
	def merge(self):
		pass
	
	def get_worksheet(self):
		pass

class Xl_reader():
	def __init__(self):
		pass

	def nxt_letter(self, letter):
		return str(chr(ord(letter)+1)) # possible up to z column

	def nxt_num(self, num_str):
		return str(int(num_str)+1)

	def read_cell(self, r, c):
		a=1

	def write_in_row(self, r, c, contents_list):
		 # r, c is string
		for n in range(len(contents_list)):
			index = c + r
			self.xl.write(r, c, contents_list[n])
			c = self.nxt_letter(c)
	def write_in_col(self, r, c, contents_list):
		pass

	def open_xl_file(self, file_name):
		 # if exists already, remove it
		os.system('rm '+ file_name)
		self.xl = XL(file_name)

	def save_xl(self):
		self.xl.save_xl()

class Xl_writer():
	def __init__(self):
		pass

	def nxt_letter(self, letter):
		return str(chr(ord(letter)+1)) # possible up to z column

	def nxt_num(self, num_str):
		return str(int(num_str)+1)

	def write_in_row(self, r, c, contents_list):
		 # r, c is string
		for n in range(len(contents_list)):
			index = c + r
			self.xl.write(r, c, contents_list[n])
			c = self.nxt_letter(c)
	def write_in_col(self, r, c, contents_list):
		pass

	def open_xl_file(self, file_name):
		 # if exists already, remove it
		os.system('rm '+ file_name)
		self.xl = XL(file_name)
	
	def save_xl(self):
		self.xl.save_xl()

class Abstraction():
	def __init__(self):
		self.row = 1
		self.subcort_result_folder = '/home/sp/fsl/T1_miguel/subcort_results'
		self.subcort_result_name = ['control_20181015_1745.txt', 'patient_left_20181015_1745.txt', 'patient_right_20181015_1745.txt']
		 #self.subcort_result_name = ['control_20181005_1229.txt'.'patient_right_20181006_2027.txt','patient_left_20181006_2027.txt']
		self.xl_file_name = ['control', 'patient_left', 'patient_right']
		self.subcort_name = ['L-Thalamus-Proper ','L-Caudate ','L-Putamen ','L-Pallidum ','Brain-Stem /4th Ventricle ','L-Hippocampus ','L-Amygdala ','L-Accumbens-area','R-Thalamus-Proper ','R-Caudate ','R-Putamen ','R-Pallidum ','R-Hippocampus','R-Amygdala ','R-Accumbens-area']
		 #self.subcort_name = self.sort_list(list(self.subcort_name))
		self.subcort_name = list(self.subcort_name)
		self.no_result = []

	def sort_list(self, l):
		new = []
		for e in sorted(l):
			new.append(e)
		return new
	
	def procedure(self, index):
		self.file_reader = FileReader()
		self.file_reader.open_file(self.subcort_result_folder + '/' + self.subcort_result_name[index])
		self.contents = self.file_reader.get_all_lines()
#		for line in self.contents:
#			print(line)
		 #print(self.contents)

		self.writer = Xl_writer()
		self.writer.open_xl_file(self.xl_file_name[index] + '.xlsx')
		
		self.writer.write_in_row(self.get_row(), str('C'), self.subcort_name)
		self.write_contents()
		 # now read subcortical result file and write it on the xl file	
		
		self.writer.save_xl()
		self.print_no_result(index)
	
	def print_no_result(self, index):
		print('in {}, there is no result to these people.'.format(self.subcort_result_name[index]))
		print(self.no_result)

	def get_row(self):
		self.row = self.row+1
		return str(self.row-1)

	def is_name_pos(self, line):
		if line == '===' or line == '===\n':	return True
		else:									return False

	def is_volume_pos(self, line):
		if line == '---' or line == '---\n':	return True
		else:									return False

	def write_contents(self):
		for index in range(len(self.contents)):
			if self.is_name_pos(self.contents[index]):
				if self.is_name_pos(self.contents[index+5]):
					print('there is no stat result.')
					self.no_result.append(self.contents[index+1])
					continue
				self.write_a_person(index+1)

	def write_a_person(self, index):
		name_parse = self.contents[index].split(' ')
		name = name_parse[2]
		line = []
		line.append(name)
		for i in range(15):
			num = self.contents[index+4+i].split(' ')[0]
			line.append(num)
		self.writer.write_in_row(self.get_row(), str('B'), line)
		 #print(line)

def main():
	for index in range(3):
		proc = Abstraction()
		proc.procedure(index)
