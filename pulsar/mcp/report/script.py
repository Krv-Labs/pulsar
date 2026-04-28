"""Standalone report JavaScript."""

from __future__ import annotations


def _render_report_script() -> str:
    """Return the standalone report JS."""
    return """
    window.addEventListener('DOMContentLoaded', () => {
      const navLinks = Array.from(document.querySelectorAll('.nav-link'));
      const sections = Array.from(document.querySelectorAll('.report-section[id], .cluster-section[id]'));

      const setActiveNav = (id) => {
        navLinks.forEach((link) => {
          link.classList.toggle('is-active', link.getAttribute('href') === `#${id}`);
        });
      };

      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const id = entry.target.getAttribute('id');
            if (id) setActiveNav(id);
          }
        });
      }, { rootMargin: '-20% 0px -65% 0px', threshold: 0.05 });

      sections.forEach((section) => observer.observe(section));

      const resetGraphNodes = () => {
        document.querySelectorAll('.graph-node').forEach((node) => {
          node.setAttribute('fill-opacity', node.dataset.baseOpacity || '0.78');
          node.setAttribute('r', node.dataset.baseRadius || '2');
          node.setAttribute('fill', node.dataset.baseFill || '#2563eb');
          node.removeAttribute('stroke');
          node.removeAttribute('stroke-width');
        });
      };

      const highlightCluster = (clusterId) => {
        document.querySelectorAll('.graph-node').forEach((node) => {
          const isMatch = node.dataset.cluster === clusterId;
          if (isMatch) {
            node.setAttribute('fill-opacity', '1');
            node.setAttribute('r', String(Math.max(Number(node.dataset.baseRadius || '2'), 4.4)));
            node.setAttribute('stroke', node.dataset.baseFill || '#2563eb');
            node.setAttribute('stroke-width', '1.1');
          } else {
            node.setAttribute('fill-opacity', '0.12');
          }
        });
      };

      document.querySelectorAll('[data-cluster-id]').forEach((target) => {
        target.addEventListener('mouseenter', () => highlightCluster(target.dataset.clusterId));
        target.addEventListener('focus', () => highlightCluster(target.dataset.clusterId), true);
        target.addEventListener('mouseleave', resetGraphNodes);
        target.addEventListener('blur', resetGraphNodes, true);
      });

      const searchInput = document.getElementById('clusterSearch');
      if (searchInput) {
        searchInput.addEventListener('input', (event) => {
          const query = event.target.value.trim().toLowerCase();
          document.querySelectorAll('.cluster-card').forEach((card) => {
            const matches = card.textContent.toLowerCase().includes(query);
            card.hidden = !matches;
          });
        });
      }

      document.querySelectorAll('[data-copy-target]').forEach((button) => {
        button.addEventListener('click', async () => {
          const target = document.getElementById(button.dataset.copyTarget);
          if (!target) return;

          const text = target.value || target.textContent || '';
          try {
            if (navigator.clipboard && window.isSecureContext) {
              await navigator.clipboard.writeText(text);
            } else {
              target.focus();
              target.select();
              document.execCommand('copy');
            }
            const original = button.textContent;
            button.textContent = 'Copied';
            window.setTimeout(() => {
              button.textContent = original;
            }, 1400);
          } catch (_error) {
            target.focus();
            target.select();
          }
        });
      });
    });
    """
