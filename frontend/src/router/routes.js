import BaseView from '@/views/BaseView.vue'
import MultimodalView from '@/views/MultimodalView.vue'

const routes = [
  {
    path: '/',
    redirect: '/baseHome'
  },
  {
    path: '/baseHome',
    name: 'BaseHome',
    component: BaseView
  },
  {
    path: '/multimodal',
    name: 'Multimodal',
    component: MultimodalView
  },
  {
    path: '/compare',
    name: 'Compare',
    component: () => import('@/views/ComparisonView.vue')
  }
]

export default routes