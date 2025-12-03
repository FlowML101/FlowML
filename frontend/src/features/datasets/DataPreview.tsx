import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Database } from 'lucide-react'

// Mock Titanic dataset
const mockData = [
  { PassengerId: 1, Survived: 0, Pclass: 3, Name: 'Braund, Mr. Owen Harris', Sex: 'male', Age: 22, SibSp: 1, Parch: 0, Fare: 7.25 },
  { PassengerId: 2, Survived: 1, Pclass: 1, Name: 'Cumings, Mrs. John Bradley', Sex: 'female', Age: 38, SibSp: 1, Parch: 0, Fare: 71.28 },
  { PassengerId: 3, Survived: 1, Pclass: 3, Name: 'Heikkinen, Miss. Laina', Sex: 'female', Age: 26, SibSp: 0, Parch: 0, Fare: 7.92 },
  { PassengerId: 4, Survived: 1, Pclass: 1, Name: 'Futrelle, Mrs. Jacques Heath', Sex: 'female', Age: 35, SibSp: 1, Parch: 0, Fare: 53.10 },
  { PassengerId: 5, Survived: 0, Pclass: 3, Name: 'Allen, Mr. William Henry', Sex: 'male', Age: 35, SibSp: 0, Parch: 0, Fare: 8.05 },
  { PassengerId: 6, Survived: 0, Pclass: 3, Name: 'Moran, Mr. James', Sex: 'male', Age: null, SibSp: 0, Parch: 0, Fare: 8.46 },
  { PassengerId: 7, Survived: 0, Pclass: 1, Name: 'McCarthy, Mr. Timothy J', Sex: 'male', Age: 54, SibSp: 0, Parch: 0, Fare: 51.86 },
  { PassengerId: 8, Survived: 0, Pclass: 3, Name: 'Palsson, Master. Gosta Leonard', Sex: 'male', Age: 2, SibSp: 3, Parch: 1, Fare: 21.07 },
  { PassengerId: 9, Survived: 1, Pclass: 3, Name: 'Johnson, Mrs. Oscar W', Sex: 'female', Age: 27, SibSp: 0, Parch: 2, Fare: 11.13 },
  { PassengerId: 10, Survived: 1, Pclass: 2, Name: 'Nasser, Mrs. Nicholas', Sex: 'female', Age: 14, SibSp: 1, Parch: 0, Fare: 30.07 },
]

const columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

export function DataPreview() {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5 text-blue-500" />
              Dataset Preview
            </CardTitle>
            <CardDescription>First 10 rows of titanic.csv</CardDescription>
          </div>
          <div className="flex gap-2">
            <Badge variant="outline">891 rows</Badge>
            <Badge variant="outline">12 columns</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="border border-zinc-800 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  {columns.map((col) => (
                    <TableHead key={col} className="whitespace-nowrap">
                      {col}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {mockData.map((row) => (
                  <TableRow key={row.PassengerId}>
                    <TableCell className="font-mono text-xs">{row.PassengerId}</TableCell>
                    <TableCell>
                      <Badge variant={row.Survived === 1 ? 'success' : 'secondary'} className="text-xs">
                        {row.Survived}
                      </Badge>
                    </TableCell>
                    <TableCell>{row.Pclass}</TableCell>
                    <TableCell className="max-w-[200px] truncate">{row.Name}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className="text-xs capitalize">
                        {row.Sex}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-zinc-400">{row.Age || 'N/A'}</TableCell>
                    <TableCell>{row.SibSp}</TableCell>
                    <TableCell>{row.Parch}</TableCell>
                    <TableCell className="font-mono text-xs">${row.Fare.toFixed(2)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
        
        <div className="mt-4 text-xs text-zinc-500 text-center">
          Showing 10 of 891 rows
        </div>
      </CardContent>
    </Card>
  )
}
