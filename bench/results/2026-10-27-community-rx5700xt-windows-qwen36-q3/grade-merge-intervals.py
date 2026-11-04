"""Execute a bounded pure-function coding probe with a strict AST/builtin allowlist."""
import ast,copy,json,sys
from pathlib import Path
answer=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
try:code=json.loads(answer['text'])['code']
except (ValueError,KeyError,TypeError):print(json.dumps({'status':'format_failure','passed':False}));raise SystemExit(1)
tree=ast.parse(code)
denied=(ast.Import,ast.ImportFrom,ast.ClassDef,ast.With,ast.AsyncWith,ast.Try,ast.Global,ast.Nonlocal,ast.Lambda,ast.While,ast.AsyncFunctionDef,ast.Await,ast.Yield,ast.YieldFrom,ast.Delete)
allowed_calls={'sorted','list','len','min','max','range','enumerate','tuple'}
for node in ast.walk(tree):
 if isinstance(node,denied):raise ValueError('Disallowed syntax: '+type(node).__name__)
 if isinstance(node,ast.Name) and node.id.startswith('__'):raise ValueError('Dunder access rejected')
 if isinstance(node,ast.Attribute) and node.attr not in ('append','copy'):raise ValueError('Only list append/copy attributes permitted')
 if isinstance(node,ast.Call):
  if isinstance(node.func,ast.Name) and node.func.id not in allowed_calls:raise ValueError('Call rejected: '+node.func.id)
  if not isinstance(node.func,(ast.Name,ast.Attribute)):raise ValueError('Indirect call rejected')
if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!='merge_intervals':raise ValueError('Expected only merge_intervals function')
space={'__builtins__':{k:__builtins__[k] if isinstance(__builtins__,dict) else getattr(__builtins__,k) for k in allowed_calls}}
exec(compile(tree,'<model-pure-function>','exec'),space)
cases=[([],[]),([[1,3],[2,6],[8,10],[15,18]],[[1,6],[8,10],[15,18]]),([[5,7],[1,2],[2,4]],[[1,4],[5,7]]),([[1,10],[2,3],[4,8]],[[1,10]]),([[-5,-1],[-3,0],[1,1]],[[-5,0],[1,1]]),([[2,2],[2,2]],[[2,2]]),([[9,10],[1,1],[4,5]],[[1,1],[4,5],[9,10]]),([[0,1],[1,2],[2,3]],[[0,3]])]
results=[]
for value,expected in cases:
 before=copy.deepcopy(value);actual=space['merge_intervals'](value)
 results.append({'input':before,'expected':expected,'actual':actual,'input_preserved':value==before,'passed':actual==expected and value==before})
print(json.dumps({'status':'tested','passed':all(x['passed'] for x in results),'cases':results}))
raise SystemExit(0 if all(x['passed'] for x in results) else 1)
