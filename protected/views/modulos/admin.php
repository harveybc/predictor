<?php
$this->breadcrumbs=array(
	'Módulos'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Gestionar'),
);

$this->menu=array(
		array('label'=>Yii::t('app',
				'Lista de Módulos'), 'url'=>array('index')),
		array('label'=>Yii::t('app', 'Nuevo Módulo'),
				'url'=>array('create')),
			);

		Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('modulos-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
		?>

<?php $this->setPageTitle (' Gestionar&nbsp;Módulos'); ?>

<?php echo CHtml::link(Yii::t('app', 'Búsqueda Avanzada'),'#',array('class'=>'search-button')); ?>
<div class="search-form" style="display:none">
<?php $this->renderPartial('_search',array(
	'model'=>$model,
)); ?>
</div>

<?php $this->widget('zii.widgets.grid.CGridView', array(
	'id'=>'modulos-grid',
	'dataProvider'=>$model->search(),
	'filter'=>$model,
         'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
	'columns'=>array(
		//'id',
		'descripcion',
		array(
			'class'=>'CButtonColumn',
		),
	),
)); ?>
