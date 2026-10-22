# Apps

该目录用于最终可执行程序的依赖组装和用户流程。应用层可以选择具体 adapters，但不得实现检测、立体匹配、滤波或数据格式规则。

当前 NX 入口由 `adapters/legacy_nx` 硬件适配器提供，统一通过 `volleyball run` 或 `volleyball capture` 启动。待硬件代码进一步拆分后，再在此添加原生实时应用。

