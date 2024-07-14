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

可知这里的`op`的类型为`OperatorHandle`,它会保存在`operatorLookupTable_`中。

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

通过注册，形成这样的映射关系`op_name`-->`op`, `dispatch_key`+`op`--->`kernel`

`op`对应`OperatorHandle`，转换为`RegistrationHandleRAII`，放在了`registrars_`里面。

## 算子调用


