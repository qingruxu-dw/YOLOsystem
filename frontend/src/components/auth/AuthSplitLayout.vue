<template>
  <div class="auth-split-page">
    <div class="auth-split-container">
      <section class="left">
        <!-- 装饰性瞄准框 -->
        <div class="tech-deco top-left"></div>
        <div class="tech-deco top-right"></div>
        <div class="tech-deco bottom-left"></div>
        <div class="tech-deco bottom-right"></div>

        <div class="slider">
          <Transition name="slide-fade" mode="out-in">
            <div :key="activeIndex" class="slider-card">
              <div class="card-title">{{ activeCard.title }}</div>
              <div v-if="activeCard.lead" class="card-lead">{{ activeCard.lead }}</div>

              <ul v-if="activeCard.bullets?.length" class="card-list">
                <li v-for="(b, i) in activeCard.bullets" :key="i">{{ b }}</li>
              </ul>

              <div v-if="activeCard.steps?.length" class="steps">
                <div v-for="(s, i) in activeCard.steps" :key="i" class="step">
                  <div class="step-index">{{ i + 1 }}</div>
                  <div class="step-text">{{ s }}</div>
                </div>
              </div>
            </div>
          </Transition>

          <div class="slider-controls">
            <el-button size="small" plain class="tech-btn" :disabled="safeCards.length <= 1" @click="prev">PREV</el-button>
            <div class="dots">
              <button
                v-for="(_, i) in safeCards"
                :key="i"
                type="button"
                class="dot"
                :class="{ active: i === activeIndex }"
                @click="go(i)"
              />
            </div>
            <el-button size="small" plain class="tech-btn" :disabled="safeCards.length <= 1" @click="next">NEXT</el-button>
          </div>
        </div>
      </section>

      <section class="right">
        <slot />
      </section>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import logo from '@/assets/images/tubiao.png'

const props = defineProps({
  brandName: { type: String, default: '清视智监' },
  subtitle: { type: String, default: '无人机图像去雾目标检测系统' },
  logoSrc: { type: String, default: logo },
  cards: {
    type: Array,
    default: () => ([
      {
        title: '功能特点',
        lead: '面向无人机图像的去雾增强与目标检测，一站式完成。',
        bullets: [
          '智能去雾增强：提升雾霾场景可视性',
          'YOLO 目标检测：快速定位关键目标',
          '多模态引导：文本提示增强检测效果',
          '对比模式：一键评估多方案结果'
        ]
      },
      {
        title: '操作流程',
        lead: '跟随步骤完成从上传到结果查看的全流程。',
        steps: [
          '上传图像（支持常见 JPG/PNG）',
          '选择模式：基础 / 多模态 / 对比',
          '按需设置文本引导与去雾强度',
          '点击开始处理，等待生成结果',
          '查看检测结果、类别统计与可视化'
        ]
      },
      {
        title: '使用建议',
        lead: '更稳定的效果来自合理的参数与输入。',
        bullets: [
          '雾更浓时可适当提高去雾强度',
          '多模态引导建议使用英文关键词描述目标',
          '对比模式适合进行参数/方案效果评估',
          '处理耗时取决于图像尺寸与模型配置'
        ]
      }
    ])
  }
})

const fallbackCards = [
  {
    title: '功能特点',
    lead: '面向无人机图像的去雾增强与目标检测，一站式完成。',
    bullets: [
      '智能去雾增强：提升雾霾场景可视性',
      'YOLO 目标检测：快速定位关键目标',
      '多模态引导：文本提示增强检测效果',
      '对比模式：一键评估多方案结果'
    ]
  },
  {
    title: '操作流程',
    lead: '跟随步骤完成从上传到结果查看的全流程。',
    steps: [
      '上传图像（支持常见 JPG/PNG）',
      '选择模式：基础 / 多模态 / 对比',
      '按需设置文本引导与去雾强度',
      '点击开始处理，等待生成结果',
      '查看检测结果、类别统计与可视化'
    ]
  },
  {
    title: '使用建议',
    lead: '更稳定的效果来自合理的参数与输入。',
    bullets: [
      '雾更浓时可适当提高去雾强度',
      '多模态引导建议使用英文关键词描述目标',
      '对比模式适合进行参数/方案效果评估',
      '处理耗时取决于图像尺寸与模型配置'
    ]
  }
]

const safeCards = computed(() => (props.cards?.length ? props.cards : fallbackCards))
const activeIndex = ref(0)
const activeCard = computed(() => safeCards.value[Math.min(activeIndex.value, safeCards.value.length - 1)])

watch(safeCards, (next) => {
  if (!next.length) {
    activeIndex.value = 0
    return
  }
  if (activeIndex.value > next.length - 1) {
    activeIndex.value = next.length - 1
  }
})

const go = (idx) => {
  const len = safeCards.value.length
  if (!len) return
  activeIndex.value = Math.max(0, Math.min(idx, len - 1))
}

const prev = () => {
  const len = safeCards.value.length
  if (len <= 1) return
  activeIndex.value = (activeIndex.value - 1 + len) % len
}

const next = () => {
  const len = safeCards.value.length
  if (len <= 1) return
  activeIndex.value = (activeIndex.value + 1) % len
}
</script>

<style scoped>
.auth-split-page {
  min-height: calc(100vh - 64px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px 12px;
  background: radial-gradient(1200px 600px at 10% 10%, rgba(30, 128, 255, 0.12), transparent 60%),
    radial-gradient(900px 500px at 90% 30%, rgba(114, 46, 209, 0.10), transparent 60%),
    #ffffff;
}

.auth-split-container {
  width: min(960px, 100%);
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.15);
  background: #fff;
  min-height: 520px;
}

.left {
  position: relative;
  padding: 32px;
  background: linear-gradient(135deg, #0b1220 0%, #1a2332 100%);
  color: #fff;
  display: flex;
  flex-direction: column;
  justify-content: center;
  overflow: hidden;
}

/* 科技感装饰元素 */
.tech-deco {
  position: absolute;
  width: 20px;
  height: 20px;
  border: 2px solid rgba(0, 255, 255, 0.3);
  transition: all 0.3s ease;
}
.tech-deco.top-left { top: 20px; left: 20px; border-right: 0; border-bottom: 0; }
.tech-deco.top-right { top: 20px; right: 20px; border-left: 0; border-bottom: 0; }
.tech-deco.bottom-left { bottom: 20px; left: 20px; border-right: 0; border-top: 0; }
.tech-deco.bottom-right { bottom: 20px; right: 20px; border-left: 0; border-top: 0; }

.left:hover .tech-deco {
  border-color: rgba(0, 255, 255, 0.8);
  width: 24px;
  height: 24px;
}

.right {
  display: flex;
  align-items: center;
  justify-content: center;
  background: #ffffff;
  padding: 32px;
}

.slider {
  position: relative;
  z-index: 2;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.slider-card {
  padding: 0 12px;
}

.card-title {
  font-size: 24px;
  font-weight: 700;
  margin-bottom: 12px;
  background: linear-gradient(90deg, #fff, #8ec5fc);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: 1px;
}

.card-lead {
  font-size: 14px;
  line-height: 1.6;
  color: rgba(255, 255, 255, 0.7);
  margin-bottom: 24px;
}

.card-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.card-list li {
  position: relative;
  padding-left: 16px;
  font-size: 13px;
  color: rgba(255, 255, 255, 0.85);
  line-height: 1.5;
}

.card-list li::before {
  content: '';
  position: absolute;
  left: 0;
  top: 8px;
  width: 4px;
  height: 4px;
  background: #00ffff;
  box-shadow: 0 0 4px #00ffff;
  border-radius: 50%;
}

.steps {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.step {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.05);
  transition: all 0.3s;
}
.step:hover {
  background: rgba(255, 255, 255, 0.06);
  border-color: rgba(0, 255, 255, 0.2);
}

.step-index {
  flex-shrink: 0;
  width: 20px;
  height: 20px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
  background: rgba(0, 255, 255, 0.15);
  color: #00ffff;
  border: 1px solid rgba(0, 255, 255, 0.3);
}

.step-text {
  font-size: 13px;
  line-height: 1.5;
  color: rgba(255, 255, 255, 0.8);
}

.slider-controls {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 12px;
  padding-top: 20px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.tech-btn {
  background: transparent !important;
  border: 1px solid rgba(255, 255, 255, 0.2) !important;
  color: rgba(255, 255, 255, 0.6) !important;
  font-family: 'Courier New', Courier, monospace;
  font-size: 12px;
  letter-spacing: 1px;
}
.tech-btn:hover:not(:disabled) {
  border-color: #00ffff !important;
  color: #00ffff !important;
  box-shadow: 0 0 8px rgba(0, 255, 255, 0.2);
}

.dots {
  display: flex;
  gap: 8px;
}

.dot {
  width: 24px;
  height: 2px;
  border: 0;
  background: rgba(255, 255, 255, 0.2);
  cursor: pointer;
  padding: 0;
  transition: all 0.3s ease;
}

.dot.active {
  background: #00ffff;
  box-shadow: 0 0 4px #00ffff;
}

.slide-fade-enter-active,
.slide-fade-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.slide-fade-enter-from,
.slide-fade-leave-to {
  opacity: 0;
  transform: translateX(10px);
}

@media (max-width: 960px) {
  .auth-split-container {
    grid-template-columns: 1fr;
    max-width: 480px;
  }
  .left {
    padding: 24px;
    min-height: auto;
  }
  .tech-deco { display: none; }
}
</style>
