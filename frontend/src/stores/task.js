import { defineStore } from 'pinia'
import { uploadImage, startProcess, getTask, getDetections } from '@/utils/api'

export const useTaskStore = defineStore('task', {
  state: () => ({
    // 文件与参数
    file: null,
    inputPreviewUrl: '',
    dehazeStrength: 0.5,
    textPrompt: '',

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
    detectionSummary: [],

    // 错误信息
    errorMessage: ''
  }),
  actions: {
    setFile(file) {
      // 释放旧的预览 URL，避免内存泄漏
      if (this.inputPreviewUrl) {
        try { URL.revokeObjectURL(this.inputPreviewUrl) } catch (e) {}
      }
      this.file = file
      if (file) {
        this.inputPreviewUrl = URL.createObjectURL(file)
        this.status = 'uploaded'
      } else {
        this.inputPreviewUrl = ''
        this.status = 'idle'
      }
    },
    async upload(mode = 'normal', textPrompt = '') {
      if (!this.file) return
      this.uploading = true
      this.errorMessage = ''
      try {
        const { data } = await uploadImage(this.file, mode, textPrompt)
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
    async startProcessTask({ mode = 'normal', textPrompt = '' } = {}) {
      if (!this.file) return
      this.processing = true
      this.status = 'processing'
      this.errorMessage = ''
      try {
        const { data } = await uploadImage(this.file, mode, textPrompt)
        this.dehazeImageUrl = data?.dehazed || ''
        this.detectImageUrl = data?.detected || ''
        // 解析 basename：来自 original 路径 /api/image/{timestamp}/1_original_{basename}.jpg
        let basename = ''
        const original = data?.original || ''
        const m = original.match(/1_original_(.+)\.jpg$/)
        if (m && m[1]) basename = m[1]
        // 拉取检测结果列表
        try {
          if (data?.timestamp && basename) {
            const resp = await getDetections(data.timestamp, basename)
            const list = resp?.data?.list || []
            this.detections = list
            const map = {}
            for (const item of list) {
              const k = item?.label || ''
              if (!k) continue
              map[k] = (map[k] || 0) + 1
            }
            this.detectionSummary = Object.keys(map).map(k => ({ label: k, count: map[k] }))
          } else {
            this.detections = []
            this.detectionSummary = []
          }
        } catch (e) {
          this.detections = []
          this.detectionSummary = []
        }
        this.status = 'done'
        this.processing = false
      } catch (err) {
        this.status = 'error'
        this.processing = false
        this.errorMessage = err?.response?.data?.error || err.message || '处理失败'
      }
    },
    startPolling() {
      /* 后端直接返回处理结果，暂不需要轮询 */
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
      if (this.inputPreviewUrl) {
        try { URL.revokeObjectURL(this.inputPreviewUrl) } catch (e) {}
      }
      this.inputPreviewUrl = ''
      this.dehazeStrength = 0.5
      this.textPrompt = ''
      this.taskId = ''
      this.status = 'idle'
      this.uploading = false
      this.processing = false
      this.dehazeImageUrl = ''
      this.detectImageUrl = ''
      this.detections = []
      this.detectionSummary = []
      this.errorMessage = ''
    }
  }
})
