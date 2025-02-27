export const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const ENDPOINTS = {
    SEARCH: '/search',
    TENDER_DETAILS: (id: string) => `/tender/${id}`,
};

export const REGIONS = [
  { code: '77', name: 'Москва' },
  { code: '78', name: 'Санкт-Петербург' },
  // Добавьте другие регионы по необходимости
]; 