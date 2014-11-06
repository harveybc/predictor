<?php
$this->breadcrumbs=array(
	'Archivoses'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Manage'),
);

$this->menu=array(
		array('label'=>Yii::t('app',
				'List Archivos'), 'url'=>array('index')),
		array('label'=>Yii::t('app', 'Create Archivos'),
				'url'=>array('create')),
			);

		Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('archivos-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
		?>

<?php $this->setPageTitle (' Manage&nbsp;Archivoses'); ?>

<?php echo CHtml::link(Yii::t('app', 'Advanced Search'),'#',array('class'=>'search-button')); ?>
<div class="search-form" style="display:none">
<?php $this->renderPartial('_search',array(
	'model'=>$model,
)); ?>
</div>

<?php $this->widget('zii.widgets.grid.CGridView', array(
	'id'=>'archivos-grid',
	'dataProvider'=>$model->search(),
	'filter'=>$model,
	'columns'=>array(
		'id',
	        array(
            'header'=>'Nombre',
            'type'=>'raw',
            'value' => 'CHTML::link($data->nombre,"/index.php/archivos/displayArchivo?id=".$data->id)'
        ),
		'tipo',
		'tamano',
            
		//'contenido',
		array(
			'class'=>'CButtonColumn',
		),
	),
)); ?>
