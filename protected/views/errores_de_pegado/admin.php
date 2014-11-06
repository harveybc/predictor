<?php
$this->breadcrumbs=array(
	'Errores De Pegados'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Manage'),
);

$this->menu=array(
		array('label'=>Yii::t('app',
				'List Errores_de_pegado'), 'url'=>array('index')),
		array('label'=>Yii::t('app', 'Create Errores_de_pegado'),
				'url'=>array('create')),
			);

		Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('errores-de-pegado-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
		?>

<?php $this->setPageTitle (' Manage&nbsp;Errores De Pegados'); ?>

<?php echo CHtml::link(Yii::t('app', 'Advanced Search'),'#',array('class'=>'search-button')); ?>
<div class="search-form" style="display:none">
<?php $this->renderPartial('_search',array(
	'model'=>$model,
)); ?>
</div>

<?php $this->widget('zii.widgets.grid.CGridView', array(
	'id'=>'errores-de-pegado-grid',
	'dataProvider'=>$model->search(),
	//'filter'=>$model,
        'cssFile'=>'/themes/gridview/styles.css',
	'columns'=>array(
		'id',
		'Campo0',
		'Campo1',
		'Campo2',
		'Campo3',
		'Campo4',
		/*
		'Campo5',
		'Campo6',
		'Campo7',
		'Campo8',
		'Campo9',
		'Campo10',
		'Campo11',
		'Campo12',
		'Campo13',
		'Campo14',
		*/
		array(
			'class'=>'CButtonColumn',
		),
	),
)); ?>
