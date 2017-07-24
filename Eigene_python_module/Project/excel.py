import xlwt
workbook=xlwt.Workbook()# encoding='utf-8'
booksheet=workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
DATA=(('1','2','3','4','5','6'),
      ('1001','A','11','m','12'),
      ('1002','B','12','f','22'),
      ('1003','C','13','f','32'),
      ('1004','D','14','m','52'),
      )
for i,row in enumerate(DATA):
    for j,col in enumerate(row):
        booksheet.write(i,j,col)
workbook.save('d:\\grade.xls')