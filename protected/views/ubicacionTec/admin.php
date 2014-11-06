<?php
$this->breadcrumbs=array(
	'Ubicación Técnica'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Gestionar'),
);

$this->menu=array(
		array('label'=>Yii::t('app',
				'Lista de Ubicación Técnica'), 'url'=>array('index')),
		array('label'=>Yii::t('app', 'Nueva Ubicación Técnica'),
				'url'=>array('create')),
			);

		Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('ubicacion-tec-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
		?>

<?php $this->setPageTitle (' Gestionar&nbsp;Ubicación Técnica'); ?>

<?php echo CHtml::link(Yii::t('app', 'Búsqueda Avanzada'),'#',array('class'=>'search-button')); ?>
<div class="search-form" style="display:none">
<?php $this->renderPartial('_search',array(
	'model'=>$model,
)); ?>
</div>

<?php $this->widget('zii.widgets.grid.CGridView', array(
	'id'=>'ubicacion-tec-grid',
	'dataProvider'=>$model->search(),
	'filter'=>$model,
        'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
	'columns'=>array(
		//'id',
		'codigoSAP',
		'descripcion',
		'padre',
		'supervisor',
		array(
			'class'=>'CButtonColumn',
		),
	),
)); ?>
