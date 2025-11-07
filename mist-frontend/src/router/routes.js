import BaseView from '@/views/BaseView.vue'

const routes = [
  {
    path: '/',
    redirect: '/baseHome'
  },
  {
    path: '/baseHome',
    name: 'BaseHome',
    component: BaseView
  }
]

export default routes