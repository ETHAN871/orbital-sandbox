# Orbital Sandbox

一个 2D 重力场沙盒小游戏，灵感来自 App Store 上的 *Orbital*。在浏览器里放置行星和黑洞，调整质量、半径和电荷，观察轨道演化。

**纯静态网页**——没有后端，没有构建步骤，直接用 GitHub Pages 部署。

## 在线游玩

把仓库 push 到 GitHub 并启用 Pages（见下文），就可以在 `https://<your-username>.github.io/<repo>/` 玩。

## 本地预览

随便一个静态服务器都能跑（ES modules 需要 HTTP，不能 file://）：

```bash
# 任选其一
python -m http.server 8000
npx serve .
```

然后浏览器打开 `http://localhost:8000`。

## 玩法

- **放置星体**：在画布上按住鼠标 → 拖出方向和速度 → 释放。拖动时显示弹弓线 + 5 秒预测虚线。
- **类型**：行星（不消除碰撞物）或黑洞（吞噬碰到的所有物体）。
- **电荷**：`+1` 引力 / `0` 中性（不施力但仍受力）/ `-1` 斥力。
- **质量 / 半径**：滑条调整。
- **轨迹长度**：滑条调整历史轨迹保留的点数（0 = 不显示）。
- **时间流速**：0 = 暂停，最高 3x。
- **暂停按钮**：在 0 和上次非零流速之间切换。
- **编辑模式**：开启后时间自动放慢，点击实体即可调整它的属性而非放置新实体。
- **清空沙盘**：一键移除全部实体。

颜色规则：行星 = 随机色；黑洞引力/中性 = 黑色；黑洞斥力（−1）= 白色。

## 部署到 GitHub Pages

1. 将本目录推送到 GitHub 仓库（默认分支 `main`）。
2. 进入仓库 **Settings → Pages**。
3. **Source** 选 **GitHub Actions**。
4. 推送任意提交到 `main`，Actions 会自动部署。完成后页面 URL 会显示在 workflow 摘要里。

workflow 已包含在 `.github/workflows/deploy.yml`。

## 技术栈

- HTML5 Canvas 2D
- 原生 ES Modules（无构建工具，无依赖）
- Velocity Verlet 数值积分（轨道稳定性优于显式 Euler）

## 项目结构

```
orbital-sandbox/
├── index.html
├── styles.css
├── src/
│   ├── main.js          # RAF 主循环 + 启动
│   ├── state.js         # 全局状态 + 物理/UI 常量
│   ├── entities.js      # 实体工厂、颜色规则
│   ├── physics.js       # 力计算、Verlet 积分、碰撞、预测
│   ├── renderer.js      # Canvas 渲染层
│   ├── input.js         # 鼠标交互（放置 / 编辑）
│   └── ui.js            # 滑条 + 按钮绑定
└── .github/workflows/deploy.yml
```
