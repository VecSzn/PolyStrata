import re,subprocess,sys,json
def code_check(text,kind):
    blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',text,re.S)
    code=blocks[0] if blocks else text.strip()
    runner=r'''
import ast,json,sys,resource
resource.setrlimit(resource.RLIMIT_CPU,(3,3))
resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
d=json.load(sys.stdin);tree=ast.parse(d['code'])
for n in ast.walk(tree):
 if isinstance(n,(ast.Import,ast.ImportFrom)):raise ValueError('imports forbidden by task')
 if isinstance(n,ast.Attribute) and n.attr.startswith('__'):raise ValueError('introspection rejected')
 if isinstance(n,ast.Name) and n.id in ['open','eval','exec','compile','globals','locals','__import__']:raise ValueError('unsafe builtin')
safe={k:__builtins__.__dict__[k] for k in ['len','list','dict','tuple','set','str','int','float','bool','sorted','range','enumerate','zip','min','max','sum','abs','any','all','isinstance','ValueError','TypeError']}
ns={'__builtins__':safe};exec(compile(tree,'generated','exec'),ns)
if d['kind']=='merge':
 cases=[([],[]),([(1,3),(2,6),(8,10),(10,12)],[(1,6),(8,12)]),([(4,5),(1,10),(2,3)],[(1,10)]),([(1,1),(1,1),(2,2)],[(1,1),(2,2)]),([(-5,-1),(-2,0),(3,4)],[(-5,0),(3,4)])]
 for inp,expected in cases:
  original=list(inp);actual=ns['merge_intervals'](inp)
  assert [tuple(x) for x in actual]==expected,(actual,expected)
  assert inp==original,'input mutated'
else:
 cases=[('',{}),(' a = 1\nb=two=three\n#comment\ninvalid\na=last\n =bad',{'a':'last','b':'two=three'}),(' x = \n# k=v\n ü = été ',{'x':'','ü':'été'})]
 for inp,expected in cases:assert ns['parse_config'](inp)==expected
print('PASS')
'''
    r=subprocess.run([sys.executable,'-I','-S','-c',runner],input=json.dumps({'code':code,'kind':kind}),text=True,capture_output=True,timeout=6)
    return r.returncode==0,(r.stdout+r.stderr)[-1200:]
