import { defineStore } from 'pinia'
import { uploadImage, startProcess, getTask } from '@/utils/api'

export const useTaskStore = defineStore('task', {
  state: () => ({
    // 文件与参数
    file: null,
    inputPreviewUrl: '',
    dehazeStrength: 0.5,

    // 任务与状态
    taskId: '',
    status: 'idle', // idle | uploaded | processing | done | error
    uploading: false,
    processing: false,
    pollingTimer: null,

    // 结果展示
    dehazeImageUrl: '',
    detectImageUrl: '',
    detections: [],

    // 错误信息
    errorMessage: ''
  }),
  actions: {
    setFile(file) {
      this.file = file
      if (file) {
        this.inputPreviewUrl = URL.createObjectURL(file)
        this.status = 'uploaded'
      } else {
        this.inputPreviewUrl = ''
        this.status = 'idle'
      }
    },
    async upload() {
      if (!this.file) return
      this.uploading = true
      this.errorMessage = ''
      try {
        const { data } = await uploadImage(this.file)
        // 可选：后端返回上传后的资源ID或路径
        // 这里保持前端预览即可
        this.status = 'uploaded'
        return data
      } catch (err) {
        this.status = 'error'
        this.errorMessage = err?.response?.data?.message || err.message || '上传失败'
        throw err
      } finally {
        this.uploading = false
      }
    },
    async startProcessTask({ mode = 'basic', textPrompt = '' } = {}) {
      if (!this.file) return
      this.processing = true
      this.status = 'processing'
      this.errorMessage = ''
      try {
        const { data } = await startProcess({
          mode,
          dehaze_strength: this.dehazeStrength,
          text_prompt: textPrompt
        })
        this.taskId = data?.task_id || data?.id || ''
        // 开始轮询查询结果
        this.startPolling()
      } catch (err) {
        this.status = 'error'
        this.errorMessage = err?.response?.data?.message || err.message || '提交处理失败'
      }
    },
    startPolling() {
      if (!this.taskId) return
      if (this.pollingTimer) clearInterval(this.pollingTimer)
      this.pollingTimer = setInterval(async () => {
        try {
          const { data } = await getTask(this.taskId)
          const state = data?.status || data?.state
          if (state === 'done' || state === 'completed') {
            // 假设后端返回结果图地址与检测列表
            this.dehazeImageUrl = data?.results?.dehaze_image_url || ''
            this.detectImageUrl = data?.results?.detect_image_url || ''
            this.detections = data?.results?.detections || []
            this.status = 'done'
            this.processing = false
            this.stopPolling()
          } else if (state === 'failed') {
            this.status = 'error'
            this.processing = false
            this.errorMessage = data?.message || '处理失败'
            this.stopPolling()
          } else {
            // 处理中，保持状态
            this.status = 'processing'
          }
        } catch (err) {
          // 轮询过程中出错，继续下一次轮询或给出提示
          // 可选：设置一个失败计数后停止
        }
      }, 1500)
    },
    stopPolling() {
      if (this.pollingTimer) {
        clearInterval(this.pollingTimer)
        this.pollingTimer = null
      }
    },
    reset() {
      this.stopPolling()
      this.file = null
      this.inputPreviewUrl = ''
      this.dehazeStrength = 0.5
      this.taskId = ''
      this.status = 'idle'
      this.uploading = false
      this.processing = false
      this.dehazeImageUrl = ''
      this.detectImageUrl = ''
      this.detections = []
      this.errorMessage = ''
    }
  }
})