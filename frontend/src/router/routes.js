import BaseView from '@/views/BaseView.vue'
import MultimodalView from '@/views/MultimodalView.vue'

const routes = [
  {
    path: '/',
    redirect: '/baseHome'
  },
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/LoginView.vue'),
    meta: { guestOnly: true }
  },
  {
    path: '/register',
    name: 'Register',
    component: () => import('@/views/RegisterView.vue'),
    meta: { guestOnly: true }
  },
  {
    path: '/baseHome',
    name: 'BaseHome',
    component: BaseView,
    meta: { requiresAuth: true }
  },
  {
    path: '/multimodal',
    name: 'Multimodal',
    component: MultimodalView,
    meta: { requiresAuth: true }
  },
  {
    path: '/compare',
    name: 'Compare',
    component: () => import('@/views/ComparisonView.vue'),
    meta: { requiresAuth: true }
  }
]

export default routes
