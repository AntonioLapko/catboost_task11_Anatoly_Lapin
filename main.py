import pandas as pd
import numpy as np
laboratory_work_list = []
laboratory_work = ("удовлетворительно", "хорошо", "отлично")
grant = np.random.randint(0, 100, (2000, 6))
student_scores = pd.DataFrame (grant, columns = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5', 'subject6'])
for i in range (2000): laboratory_work_list.insert (i, np.random.choice(laboratory_work))
student_scores['laboratory_work'] = laboratory_work_list
print(student_scores)

print(laboratory_work)