<?php
$this->breadcrumbs=array(
	'Aislamiento Fase'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Gestionar'),
);

$this->menu=array(
		array('label'=>Yii::t('app',
				'Lista de Mediciones'), 'url'=>array('index')),
		array('label'=>Yii::t('app', 'Nueva Medición'),
				'url'=>array('create')),
			);

		Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('aislamiento-fases-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
		?>

<?php $this->setPageTitle (' Gestionar&nbsp;Medición Aislamiento Fase'); ?>

<?php echo CHtml::link(Yii::t('app', 'Búsqueda Avanzada'),'#',array('class'=>'search-button')); ?>
<div class="search-form" style="display:none">
<?php $this->renderPartial('_search',array(
	'model'=>$model,
)); ?>
</div>

<?php $this->widget('zii.widgets.grid.CGridView', array(
	'id'=>'aislamiento-fases-grid',
	'dataProvider'=>$model->search(),
	//'filter'=>$model,
         'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
	'columns'=>array(
		'Toma',
		'TAG',
		'Fecha',
		'A050',
		'A1',
		'B050',
		/*
		'B1',
		'C050',
		'C1',
		'OT',
		*/
		array(
			'class'=>'CButtonColumn',
		),
	),
)); ?>
