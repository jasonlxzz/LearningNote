# 算子分发

## 算子注册

通过宏`TORCH_LIBRARY_IMPL`进行算子注册，ns表示命令空间，k表示c10::DispatcherKey，m是torch::Library。下面的两个宏来自文件`torch/library.h`

```cpp
#define TORCH_LIBRARY_IMPL(ns, k, m) _TORCH_LIBRARY_IMPL(ns, k, m, C10_UID)
```

具体的宏展开如下：

```cpp
#define _TORCH_LIBRARY_IMPL(ns, k, m, uid)                                \
  static void C10_CONCATENATE(                                            \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library&);       \
  static const torch::detail::TorchLibraryInit C10_CONCATENATE(           \
      TORCH_LIBRARY_IMPL_static_init_##ns##_##k##_, uid)(                 \
      torch::Library::IMPL,                                               \
      (c10::impl::dispatch_key_allowlist_check(c10::DispatchKey::k)       \
           ? &C10_CONCATENATE(TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid) \
           : [](torch::Library&) -> void {}),                             \
      #ns,                                                                \
      c10::make_optional(c10::DispatchKey::k),                            \
      __FILE__,                                                           \
      __LINE__);                                                          \
  void C10_CONCATENATE(                                                   \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library & m)
```

结合下面的实际例子，走一遍注册的过程

```cpp
TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("_assert_async",
TORCH_FN(wrapper_CPU___assert_async));
}
```

1）**声明**一个静态函数 `static void C10_CONCATENATE(  
TORCH_LIBRARY_IMPL_init_##ns##*##k##*, uid)(torch::Library&);`，用来注册算子。返回值为void，入参为torch::Library引用。

2）定义一个静态const对象，类型为`torch::detail::TorchLibraryInit`，变量名为`C10_CONCATENATE(  
TORCH_LIBRARY_IMPL_static_init_##ns##*##k##*, uid)`，构造入参分别为库类型枚举`torch::Library::IMPL`，上面声明的函数指针，命令空间字符串，DispatcherKey::k，后面是文件名和行号。`TorchLibraryInit`定义如下：

```cpp
class TorchLibraryInit final {
 private:
  using InitFn = void(Library&);
  Library lib_;

 public:
  TorchLibraryInit(
      Library::Kind kind,
      InitFn* fn,
      const char* ns,
      c10::optional<c10::DispatchKey> k,
      const char* file,
      uint32_t line)
      : lib_(kind, ns, k, file, line) {
    fn(lib_);
  }
};
```

其构造函数会通过`kind/ns/k/file/line`参数构造`Library lib_`，再调用fn函数。

fn函数声明过了，还差其定义。定义了静态变量后，紧接着就是fn函数的定义了，函数体位于宏使用的时候，

```cpp
m.impl("_assert_async",
TORCH_FN(wrapper_CPU___assert_async)
```

可知具体细节存在于`torch::Library`类当中

```cpp
  template <typename Name, typename Func>
  Library& impl(
      Name name,
      Func&& raw_f,
      _RegisterOrVerify rv = _RegisterOrVerify::REGISTER) & {
    // TODO: need to raise an error when you impl a function that has a
    // catch all def
#if defined C10_MOBILE
    CppFunction f(std::forward<Func>(raw_f), NoInferSchemaTag());
#else
    CppFunction f(std::forward<Func>(raw_f));
#endif
    return _impl(name, std::move(f), rv);
  }
```

首先将核函数`wrapper_CPU___assert_async`，转换为CppFunction，再调用`_impl`。

其中CppFunction初始化包含`KernelFunction func_`， `CppSignature cpp_signature_`与`FunctionSchema schema_`。

```cpp
  /// This overload accepts compile time function pointers, e.g.,
  /// `CppFunction(TORCH_FN(add_impl))`
  template <typename FuncPtr>
  explicit CppFunction(
      FuncPtr f,
      std::enable_if_t<
          c10::is_compile_time_function_pointer<FuncPtr>::value,
          std::nullptr_t> = nullptr)
      : func_(c10::KernelFunction::makeFromUnboxedFunction(f)),
        cpp_signature_(
            c10::impl::CppSignature::make<typename FuncPtr::FuncType>()),
        schema_(c10::detail::inferFunctionSchemaFromFunctor<
                typename FuncPtr::FuncType>()),
        debug_() {} 
   private:
  c10::optional<c10::DispatchKey> dispatch_key_;
  c10::KernelFunction func_;
  c10::optional<c10::impl::CppSignature> cpp_signature_;
  std::unique_ptr<c10::FunctionSchema> schema_;
  std::string debug_;
```

传给CppFunction的实际为下面的`CompileTimeFunctionPointer`，按照这个命名，它应该是编译时期可以确定类型的函数指针。

```cpp
//c10\core\CompileTimeFunctionPointer.h
#define TORCH_FN_TYPE(func)                                           \
  ::c10::CompileTimeFunctionPointer<                                  \
      std::remove_pointer_t<std::remove_reference_t<decltype(func)>>, \
      func>
#define TORCH_FN(func) TORCH_FN_TYPE(func)() 


template <class FuncType_, FuncType_* func_ptr_>
struct CompileTimeFunctionPointer final {
  static_assert(
      guts::is_function_type<FuncType_>::value,
      "TORCH_FN can only wrap function types.");
  using FuncType = FuncType_;

  static constexpr FuncType* func_ptr() {
    return func_ptr_;
  }
};
```

继续看`_impl`函数的实现，第一个参数是`operator`名称，第二个参数是`CppFunction`，第三个参数是`v = _RegisterOrVerify::REGISTER`

```cpp
Library& Library::_impl(const char* name_str, CppFunction&& f, _RegisterOrVerify rv) & {
  at::OperatorName name = _parseNameForLib(name_str);
  //some check ....

  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  switch (rv) {
    case _RegisterOrVerify::REGISTER:
      registrars_.emplace_back(
        c10::Dispatcher::singleton().registerImpl(
          std::move(name),
          dispatch_key,
          std::move(f.func_),
          f.cpp_signature_,
          std::move(f.schema_),
          debugString(std::move(f.debug_), file_, line_)
        )
      );
      break;
    case _RegisterOrVerify::VERIFY:
      c10::Dispatcher::singleton().waitForImpl(name, dispatch_key);
      break;
  }
  return *this;
}
```

首先根据字符串获取到算子名OperatorName，  然后获取dispatch_key，根据上述步骤，这里的key使用的Library的成员变量，也就是宏定义中的k，再往Library的成员变量`std::vector<c10::RegistrationHandleRAII> registrars_;`中添加注册指针。

到这里先梳理一遍注册的设计流程，然后再继续看更细节的东西。从实现上看，通过算子注册宏`TORCH_LIBRARY_IMPL`，通过`命令空间`,`DispatcherKey`等信息创建出类型为`TorchLibraryInit`的静态变量，这个静态变量中有个Library类型的成员变量`lib_`，`lib_`中成员变量`registrars_`保存了所有注册句柄。

接下来在深入句柄的创建过程，可以看到是调用了`Dispatcher`的`registerImpl`方法。

```cpp
//aten\src\ATen\core\dispatch\Dispatcher.cpp
RegistrationHandleRAII Dispatcher::registerImpl(
  OperatorName op_name,
  c10::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  c10::optional<impl::CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  std::lock_guard<std::mutex> lock(guard_->mutex);

  auto op = findOrRegisterName_(op_name);

  auto handle = op.operatorDef_->op.registerKernel(
    *this,
    dispatch_key,
    std::move(kernel),
    std::move(cpp_signature),
    std::move(inferred_function_schema),
    std::move(debug)
  ); 
  ++op.operatorDef_->def_and_impl_count;

  cond_var_.notify_all();

  return RegistrationHandleRAII([guard = this->guard_, this, op, op_name, dispatch_key, handle] {
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    deregisterImpl_(op, op_name, dispatch_key, handle);
  });
}
```

首先根据`op_name`查找或者注册`op`，`findOrRegisterName_`函数实现如下：

```cpp
//aten\src\ATen\core\dispatch\Dispatcher.cpp
OperatorHandle Dispatcher::findOrRegisterName_(const OperatorName& op_name) {
  const auto found = findOp(op_name);
  if (found != c10::nullopt) {
    return *found;
  }

  operators_.emplace_back(OperatorName(op_name));
  OperatorHandle handle(--operators_.end());
  operatorLookupTable_.write([&] (ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) {
    operatorLookupTable.emplace(op_name, handle);
  });

  return handle;
}
```

可知这里的`op`的类型为`OperatorHandle`,它会保存在`operatorLookupTable_`中，它是`Dispatcher`单例的成员变量。

有了`op`之后，调用`registerKernel`方法，绑定`dispatch_key`，`kernel`,`cpp_signature`,`inferred_function_schema`。`registerKernel`函数实现如下:

```cpp
//aten\src\ATen\core\dispatch\OperatorEntry.cpp
OperatorEntry::AnnotatedKernelContainerIterator OperatorEntry::registerKernel(
  const c10::Dispatcher& dispatcher,
  c10::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  c10::optional<CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
)
{
  auto& k = dispatch_key.has_value() ? kernels_[*dispatch_key] : kernels_[DispatchKey::CompositeImplicitAutograd];
  k.emplace_front(std::move(kernel), std::move(inferred_function_schema), std::move(debug));
  // update the dispatch table, i.e. re-establish the invariant
  // that the dispatch table points to the newest kernel
  if (dispatch_key.has_value()) {
    updateDispatchTable_(dispatcher, *dispatch_key);
  } else {
    updateDispatchTableFull_(dispatcher);
  }
  return inserted;
}
```

`registerKernel`在`OperatorEntry`类中，仅拷贝了核心的几行代码，主要是根据`dispatch_key`在`kernels_`中查找`k`，`k`找到之后将`kernel`,`inferred_function_schema`信息与其绑定。`kernels_`的结构如下，它是一个`map`表。

```cpp
  ska::flat_hash_map<DispatchKey,
#ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
                     // On mobile, we needn't worry about Jupyter notebooks.
                     std::array<AnnotatedKernel, 1>
#else
                     std::list<AnnotatedKernel>
#endif
                     > kernels_;
```

键为`DispatchKey`，值为`std::list<AnnotatedKernel>`。

通过注册，可以`Dispatcher`单例的`operatorLookupTable`中，通过`op_name`找到对应的`OperatorHandle op`,在利用 `dispatch_key`信息找到对应的`kernel`函数实现。

同时`op`对应`OperatorHandle`，转换为`RegistrationHandleRAII`，放在了`registrars_`里面。

## 算子调用

当初步了解算子的注册流程之后，那么

1）具体是如何从`python`层一级一级的调用的？

2）`box`和`unbox`机制到底在什么地方用到，应用场景是什么

3）这样的调用的好处在哪里？

带着这三个问题，理一下`nn.Conv2d`的前向调用。

python端的调用来自于

```python
#torch\nn\modules\conv.py
from .. import functional as F

class Conv2d(_ConvNd):
    def __init__(self,....):
        #......
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
```

这个`python`前端代码都原始工程里面都能见到，可以看到它调用了`F.conv2d`函数。

`F`即`functional.py`，看到这个文件中`conv2d`在`torch.conv2d`上加了一些注释和使用用例。

```python
#torch\nn\functional.py
conv2d = _add_docstr(
    torch.conv2d,
#annotation and example 
)
```

到这里几乎就已经蒙了，因为我们根本不知道`torch.con2d`在什么地方。不过肯定是在`import`的时候添加到`torch`模块中，即这一切都发生在`torch/__init__.py`。

让我们快速回顾一下在`python`中建立`C/C++`扩展，即在`python`中如何调用`C/C++`函数。

看看`chatgpt`生成的例子：`example.cpp`提供了模块初始化函数`PyInit_example`，`example`是模块名，以及模块的定义和方法的定义。

```cpp
//example.cpp
#include <Python.h>

// Example function that adds two integers
static PyObject* add(PyObject* self, PyObject* args) {
    int a, b;
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL;
    }
    return PyLong_FromLong(a + b);
}

// Method definitions
static PyMethodDef ExampleMethods[] = {
    {"add", add, METH_VARARGS, "Add two integers"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef examplemodule = {
    PyModuleDef_HEAD_INIT,
    "example",
    "Example module that adds two integers",
    -1,
    ExampleMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_example(void) {
    return PyModule_Create(&examplemodule);
}


```

然后再提供一个`setup.py`

```python
from setuptools import setup, Extension

# Define the extension module
example_module = Extension(
    'example',         # Name of the module
    sources=['example.cpp'],  # Source file(s)
    language='c++'     # Specify C++ language
)

# Run the setup
setup(
    name='example',
    version='1.0',
    description='Example C++ extension module',
    ext_modules=[example_module]
)

```

调用这个脚本可以自动帮你把需要的`so`编译出来。

```python
python setup.py build
```

或者直接进行安装

```python
python setup.py install
```

这个例子已经很清晰，小结一下

- `PyObject *PyModule_Create(PyModuleDef *def)`，创建模块

- `PyModuleDef` 定义模块

- `PyMethodDef` 定义模块方法

利用`setuptools`即可快速定义模块和编译。

有了这些背景知识后，再回过头来`torch/__init__.py`，猜测就是在该文件中将`conv2d`方法定义，或者绑定到`torch`模块。乍一看`__init__.py`文件其实也看不出什么东西，我这里是从底层往上看，这依赖于有个源码编译的`torch`工程，因为这部分代码都是自动生成的，源码中只提供了代码的模版。

这部分的代码生成在目录`/pytorch/torch/csrc/autograd/generated`，

搜索一下可以发现在`python_torch_functions_1.cpp`文件中，有关于卷积的代码

```cpp

static PyObject * THPVariable_conv2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymIntArrayRef[2] stride=1, SymIntArrayRef[2] padding=0, SymIntArrayRef[2] dilation=1, SymInt groups=1)",
    "conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymIntArrayRef[2] stride=1, c10::string_view padding=\"valid\", SymIntArrayRef[2] dilation=1, SymInt groups=1)",
  }, /*traceable=*/false);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1) -> Tensor

      auto dispatch_conv2d = [](const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, c10::SymInt groups) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::conv2d_symint(input, weight, bias, stride, padding, dilation, groups);
      };
      return wrap(dispatch_conv2d(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toSymInt(6)));
    }
    case 1: {
      // aten::conv2d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, str padding="valid", SymInt[2] dilation=1, SymInt groups=1) -> Tensor

      auto dispatch_conv2d = [](const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::string_view padding, c10::SymIntArrayRef dilation, c10::SymInt groups) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::conv2d_symint(input, weight, bias, stride, padding, dilation, groups);
      };
      return wrap(dispatch_conv2d(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.stringView(4), _r.symintlist(5), _r.toSymInt(6)));
    }
  }
static PyMethodDef torch_functions_shard[] = {
  //...
  {"conv2d", castPyCFunctionWithKeywords(THPVariable_conv2d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  //....
};

void gatherTorchFunctions_1(std::vector<PyMethodDef> &torch_functions) {
  constexpr size_t num_functions = sizeof(torch_functions_shard) / sizeof(torch_functions_shard[0]);
  torch_functions.insert(
    torch_functions.end(),
    torch_functions_shard,
    torch_functions_shard + num_functions);
}


```

关键的函数就是`at::conv2d_symint`，这个会函数会触发`Dispatcher`机制，最后调用底层实现（稍后会有解释细节）。先看上层的调用，`gatherTorchFunctions_1`函数，它会将所有的将当前文件列出来的所有`PyMethodDef`都进行收集到`torch_functions`中。继续查找调用`gatherTorchFunctions_1`的地方，发现调用发生在文件`torch\csrc\autograd\python_torch_functions_manual.cpp`

```cpp
void initTorchFunctions(PyObject* module) {
  static std::vector<PyMethodDef> torch_functions;
  gatherTorchFunctions(torch_functions);
  THPVariableFunctions.tp_methods = torch_functions.data();

  if (PyType_Ready(&THPVariableFunctions) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPVariableFunctions);

  // Steals
  Py_INCREF(&THPVariableFunctions);
  if (PyModule_AddObject(
          module,
          "_VariableFunctionsClass",
          reinterpret_cast<PyObject*>(&THPVariableFunctions)) < 0) {
    throw python_error();
  }
  // PyType_GenericNew returns a new reference
  THPVariableFunctionsModule =
      PyType_GenericNew(&THPVariableFunctions, Py_None, Py_None);
  // PyModule_AddObject steals a reference
  if (PyModule_AddObject(
          module, "_VariableFunctions", THPVariableFunctionsModule) < 0) {
    throw python_error();
  }
}

void gatherTorchFunctions(std::vector<PyMethodDef>& torch_functions) {
  constexpr size_t num_functions =
      sizeof(torch_functions_manual) / sizeof(torch_functions_manual[0]);
  torch_functions.assign(
      torch_functions_manual, torch_functions_manual + num_functions);
  // NOTE: Must be synced with num_shards in
  // tools/autograd/gen_python_functions.py
  gatherTorchFunctions_0(torch_functions);
  gatherTorchFunctions_1(torch_functions);
  gatherTorchFunctions_2(torch_functions); 
  //........
}
```

从上面的源码可以得知，torch.conv2d应该是和`initTorchFunctions`函数入参`module`模块的`_VariableFunctions`模块里面。继续查看`initTorchFunctions`调用的地方，可以看到调用发生在`torch\csrc\autograd\python_variable.cpp`

```cpp
bool THPVariable_initModule(PyObject* module) {
  //....
  torch::autograd::initTorchFunctions(module);
  torch::autograd::initTensorImplConversion(module);
  torch::utils::validate_numpy_for_dlpack_deleter_bug();
  return true;
}
```

`THPVariable_initModule`调用发生在`torch\csrc\Module.cpp`

```cpp
PyObject* initModule() {
  HANDLE_TH_ERRORS
  //.....
  static struct PyModuleDef torchmodule = {
      PyModuleDef_HEAD_INIT, "torch._C", nullptr, -1, methods.data()};
  module = PyModule_Create(&torchmodule);
  //...
  ASSERT_TRUE(THPVariable_initModule(module));
  //.... 
}
```

有了之前聊过扩展`python`，这里的代码读起来就不费力了，创建了一个`torch._C`的`python`模块，这个模块会传给`THPVariable_initModule`。`initModule`则是在`torch\csrc\stub.c`中调用。

```c
PyMODINIT_FUNC PyInit__C(void)
{
  return initModule();
}
```

比较一下`_C._VariableFunctionsClass.conv2d/_C._VariableFunctions.conv2d` 和 `torch.conv2d`，两者确实相同。

```python
import torch._C as _C
import torch
_C._VariableFunctionsClass.conv2d is torch.conv2d
#True
_C._VariableFunctions.conv2d is torch.conv2d
#True
```

接着看`torch.conv2d`是怎么和`_C.__VariableFunctionsClass.conv2d`或者`_C._VariableFunctions.conv2d`进行关联的。发生在`torch\__init__.py`文件中，

```python
for name in dir(_C._VariableFunctions):
    if name.startswith('__') or name in PRIVATE_OPS:
        continue
    obj = getattr(_C._VariableFunctions, name)
    obj.__module__ = 'torch'
    # Hide some APIs that should not be public
    if name == "segment_reduce":
        # TODO: Once the undocumented FC window is passed, remove the line bellow
        globals()[name] = obj
        name = "_" + name
    globals()[name] = obj
    if not name.startswith("_"):
        __all__.append(name)
```

此处`_C._VariableFunctions`里面的方法都加到的torch模块的`__all__`变量中，因此可以直接`import torch; torch.conv2d`。

理一下前面的调用，已经将问题1）解答了一大半：

`torch.conv2d` 

-> `torch._C._VariableFunctions.conv2d`

-> `THPVariable_conv2d`

-> `conv2d_symint`

继续看`conv2d_symint`调用里，这涉及前面的算子注册的知识，另外这部分的代码全都是在编译阶段自动生成的。






