# Wiki维护规范

## 依据

本 Wiki 按 GitHub Wiki 的工作方式维护: Wiki 可以作为独立 Git 仓库 clone 到本地，`Home.md` 是首页，`_Sidebar.md` 是侧边栏。

文档分类采用 Diataxis 思路: tutorial、how-to、reference、explanation 分开写。工程上不强行使用英文分类名，而是映射为:

- 快速开始: 新人路径。
- 操作手册: 标定、部署、录制、排障。
- 架构说明: 系统架构、实时管线、深度测量。
- 参考资料: 构建目标、配置字段、schema、工具索引、迁移索引。

参考:

- GitHub Docs: https://docs.github.com/en/communities/documenting-your-project-with-wikis/about-wikis
- GitHub Docs: https://docs.github.com/en/communities/documenting-your-project-with-wikis/adding-or-editing-wiki-pages
- Diataxis: https://diataxis.fr/

## 单一入口

- `wiki/Home.md` 是正式入口。
- `wiki/_Sidebar.md` 是正式导航。
- `docs/` 不再新增长期维护型手册，只保留历史报告、实验记录、计划和排障原始材料。
- 新增功能文档优先写 Wiki，再从 Wiki 链接必要的历史材料。

## 命名规则

- 页面名使用中文功能名，不加编号。
- 页面标题必须和文件名含义一致。
- 不创建“杂项”“开发记忆”“临时记录”这类长期页面。
- 临时计划放 `docs/plans/YYYY-MM-DD-主题.md`，完成后只把结论迁移到 Wiki。

## 页面结构

每个页面尽量包含:

- 当前有效结论。
- 适用范围。
- 命令或配置。
- 相关源码入口。
- 验证方法。
- 历史资料链接。

避免:

- 只写背景不写操作。
- 复制整段旧报告。
- 写“推荐/生产就绪”但没有日期、配置和验证命令。
- 在多个页面维护同一份命令。

## 修改触发条件

以下改动必须同步更新 Wiki:

- 改相机、触发、标定流程。
- 改运行配置默认值。
- 改实时管线调度、丢帧策略、队列策略。
- 新增或删除深度候选字段。
- 新增或删除 CMake target、工具脚本、配置文件。
- 改 CSV schema 或 baseline clip 输出格式。
- 更换模型、engine、推理后端。
- 改部署命令、构建命令或依赖版本。
- 新增性能结论。

## 提交流程

推荐本仓库内维护 `wiki/` 源文件，确认后同步到 GitHub Wiki 仓库:

```bash
git clone git@github.com:<owner>/<repo>.wiki.git ../<repo>.wiki
rsync -av --delete NX_volleyball/stereo_3d_pipeline/wiki/ ../<repo>.wiki/
cd ../<repo>.wiki
git status
git add .
git commit -m "docs: 更新双目追踪 wiki"
git push
```

如果直接在 GitHub 网页编辑 Wiki，必须再同步回本仓库的 `wiki/` 目录，避免本地源文件落后。

## Review 清单

提交文档前检查:

- 入口是否在 `_Sidebar.md`。
- 链接是否能在本地文件中打开。
- 命令是否和当前代码参数一致。
- `_Sidebar.md` 是否体现父子层级，而不是只在孤立页面新增内容。
- 性能数据是否带日期、配置和命令。
- 旧文档是否已经在 [文档迁移索引](文档迁移索引.md) 中标注去向。
- 是否避免把离线实验写成默认实时路径。
