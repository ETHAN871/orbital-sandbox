# Orbital Sandbox

浏览器里的二维重力沙盒：拖拽发射行星、放置吞噬体、编辑质量与半径、观察运动和轨迹。

**在线入口：[ethan871.github.io/orbital-sandbox](https://ethan871.github.io/orbital-sandbox/)**

这是交互玩具，不是天体力学或广义相对论计算工具。模型、误差和本次维护证据见 [维护与物理审计](docs/maintenance.md)。

## 运行与操作

```sh
python3 -m http.server 8000
```

打开 `http://localhost:8000`。必须通过 HTTP；ES modules 不能用 `file://` 直接运行。

- 在画布上按住并向后拖拽，松开后向反方向发射；虚线是冻结其他物体后的近似预测。
- 编辑模式可选中星体，修改属性、固定或删除；暂停在编辑模式中也生效。
- `+1` 是吸引源，`−1` 是排斥源，`0` 不施力但仍受力。这不是电磁电荷。
- “黑洞”是接触吞噬规则；不会增加吞噬者质量，不模拟事件视界。
- 时间滑条 `1×` 对应目标 `2` 模拟秒／真实秒，`3×` 对应 `6`；编辑时目标 `0.4×`。重负载时会减速以保持响应。
- 循环边界连接画布两端。膜是定性示意，不能据其斜率读取真实加速度。
- 页面刷新会清空场景；状态录制用于诊断，不是可导入的存档。

## 实现

纯静态 HTML/CSS/ES modules，无打包编译。WebGL 2 绘图；默认 [Rapier 2D 0.19.3](https://rapier.rs/) 处理接触，通过固定 `1/60 s` 步长的引力冲量推进。引力是带半径相关 Plummer 软化的直接全对计算。具备 WebGPU 时可使用 K1 引力内核，失败时回退 CPU。Rapier 从固定版本 CDN 加载；CDN 不可用时使用仓库内的 JS 求解器，碰撞响应与 Rapier 不完全相同。

- `src/main.js`、`simulation-clock.js`：固定步长循环及过载保护。
- `src/physics-rapier.js`：默认物理、镜像接触、吞噬和 CCD。
- `src/physics.js`、`physics-spatial-hash.js`：本地后备、预测、边界。
- `src/renderer-webgl.js`、`shaders.js`：渲染、轨迹和场图。
- `src/input.js`、`ui.js`、`state.js`：交互和状态。
- `src/physics-gpu-gravity.js`、`kernels/gravity_accel.wgsl`：可选 GPU 引力。

诊断参数：`?engine=cpu` 强制本地 JS；`?engine=rapier&backend=force-cpu` 保留 Rapier 并禁用 GPU 引力；`?aqm=off` 禁用自适应质量；`?fps=off` 隐藏帧率。

## 验证与发布

Node 24 可运行无依赖数值回归：

```sh
node --test tests/core.test.mjs
```

浏览器手动检查：通过同一个 HTTP 服务打开 `tests/potential.test.html`、`tests/rapier-acceptance.test.html?engine=rapier&backend=force-cpu` 和 `tests/physics-audit.test.html?engine=rapier&backend=force-cpu`。

完整浏览器测试用临时安装的 Playwright，不给项目加入运行时 npm 依赖：

```sh
npm install --prefix /tmp/orbital-test-tools playwright@1.64.1
/tmp/orbital-test-tools/node_modules/.bin/playwright install chromium
PLAYWRIGHT_MODULE=/tmp/orbital-test-tools/node_modules/playwright/index.mjs node tests/browser.mjs
```

可设 `BROWSER_PATH` 使用本机 Chrome；截图和结果写到忽略的 `artifacts/`。GPU 不可用时 GPU 专项会明确标记跳过，不能将其视为硬件测试通过。

GitHub Actions 在 PR 和 main 推送时执行测试。main 测试成功后，仅将 `index.html`、`styles.css`、`src/` 和版本标识部署到原有 GitHub Pages；发布 import map 按 commit 统一刷新全部模块缓存；文档、实验记录与测试不会进入网站目录。

## 历史恢复

2026-11 维护移除了非默认的 Planck、Worker、K2–K8 GPU 碰撞试验、28 张效果图及生成脚本，删除两份过时蓝图，重写陈旧的项目指引。未删除远端分支或改写历史。原始内容仍在维护前提交 `9c6b894` 中，例如：

```sh
git show 9c6b894:docs/webgpu-blueprint.md
git show 9c6b894:src/physics-planck.js
```

默认 Rapier、CPU 后备、GPU 引力及诊断录制继续保留。
